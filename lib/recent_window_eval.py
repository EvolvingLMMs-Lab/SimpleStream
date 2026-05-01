from __future__ import annotations

import copy
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ovo_constants import (
    BACKWARD_TASKS,
    REAL_TIME_TASKS,
    FORWARD_TASKS,
    BR_PROMPT_TEMPLATE,
    REC_PROMPT_TEMPLATE,
    SSR_PROMPT_TEMPLATE,
    CRR_PROMPT_TEMPLATE,
)

ALL_BR_TASKS = BACKWARD_TASKS + REAL_TIME_TASKS


class _TTFTStreamer:
    def __init__(self, start_time: float) -> None:
        self.start_time = start_time
        self.ttft_seconds: float | None = None

    def put(self, value: torch.Tensor) -> None:
        if self.ttft_seconds is None:
            self.ttft_seconds = time.perf_counter() - self.start_time

    def end(self) -> None:
        pass


@dataclass
class EvalChunk:
    frames: list[Image.Image]
    frame_timestamps: list[float]
    start_time: float
    end_time: float
    chunk_index: int
    fps: float


@dataclass
class RecentWindowResult:
    answer: str
    final_chunk_ids: list[int]
    generate_time: float
    ttft_seconds: float
    num_vision_tokens: int
    num_vision_tokens_before: int
    num_vision_tokens_after: int
    num_frames: int


class RecentWindowQAModel:
    """Minimal Qwen-VL wrapper for the recent-window recency baseline.

    Qwen2.5-VL keeps one ``<|vision_start|>...<|vision_end|>`` block per frame.
    Qwen3-VL overrides this path with the single-block cached builder.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        from transformers import AutoProcessor

        if "qwen3" in model_name.lower():
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLForConditionalGeneration as _ModelClass,
            )
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration as _ModelClass,
            )

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self._last_ttft_seconds: float = 0.0
        self._last_num_vision_tokens: int = 0
        self._last_num_vision_frames: int = 0

        proc_kwargs: dict[str, Any] = {}
        if os.environ.get("MIN_PIXELS"):
            proc_kwargs["min_pixels"] = int(os.environ["MIN_PIXELS"])
        if os.environ.get("MAX_PIXELS"):
            proc_kwargs["max_pixels"] = int(os.environ["MAX_PIXELS"])
        self.processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = str(device)

        _saved_ws = os.environ.pop("WORLD_SIZE", None)
        try:
            self.model = _ModelClass.from_pretrained(model_name, **model_kwargs)
        finally:
            if _saved_ws is not None:
                os.environ["WORLD_SIZE"] = _saved_ws

        self.model.eval()

        _hf_model = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        self._hf_model = _hf_model
        self.image_token_id = _hf_model.config.image_token_id
        self._visual = _hf_model.visual
        self._text_model = _hf_model.model
        self.merge_size = getattr(self._visual, "spatial_merge_size", 1)

        tokenizer = self.processor.tokenizer
        self._vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self._vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _get_hf_model(self):
        if hasattr(self, "_hf_model"):
            return self._hf_model
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def _get_visual_module(self):
        if hasattr(self, "_visual"):
            return self._visual
        hf_model = self._get_hf_model()
        if hasattr(hf_model, "visual"):
            return hf_model.visual
        return hf_model.model.visual

    def _get_text_model(self):
        if hasattr(self, "_text_model"):
            return self._text_model
        hf_model = self._get_hf_model()
        return hf_model.model if hasattr(hf_model, "model") else hf_model

    def _get_image_feature_model(self):
        hf_model = self._get_hf_model()
        if hasattr(hf_model, "get_image_features"):
            return hf_model
        return hf_model.model

    def _get_visual_dtype(self) -> torch.dtype:
        visual = self._get_visual_module()
        if hasattr(visual, "dtype"):
            return visual.dtype
        if hasattr(self.model, "dtype"):
            return self.model.dtype
        return torch.bfloat16

    def _flatten_vision_features(self, features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        if isinstance(features, (tuple, list)):
            if features and all(isinstance(item, torch.Tensor) for item in features):
                return torch.cat(list(features), dim=0)
            first = features[0] if features else None
            if isinstance(first, torch.Tensor):
                return first
            if isinstance(first, (tuple, list)) and first and all(isinstance(item, torch.Tensor) for item in first):
                return torch.cat(list(first), dim=0)
        raise TypeError(f"Unexpected vision feature type: {type(features)}")

    def _get_multimodal_model(self):
        multimodal_model = getattr(self.model, "model", None)
        if multimodal_model is None:
            raise TypeError("Expected a Qwen2.5-VL style multimodal model.")
        return multimodal_model

    def _infer_module_device(self, module: Any) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
        if hasattr(self.model, "device"):
            return self.model.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_visual_device(self) -> torch.device:
        return self._infer_module_device(self._get_visual_module())

    def _get_text_input_device(self) -> torch.device:
        embeddings = self._get_text_model().get_input_embeddings()
        return self._infer_module_device(embeddings)

    @torch.inference_mode()
    def _generate_from_model_inputs(self, prompt_length: int, **generate_kwargs: Any) -> str:
        """Run generation from prepared model inputs and decode only new tokens."""
        t0 = time.perf_counter()
        streamer = _TTFTStreamer(t0)
        generated_ids = self.model.generate(
            **generate_kwargs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )
        self._last_ttft_seconds = (
            streamer.ttft_seconds
            if streamer.ttft_seconds is not None
            else (time.perf_counter() - t0)
        )

        trimmed = [
            generated_ids[0][prompt_length:]
            if generated_ids.shape[1] > prompt_length
            else generated_ids[0]
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    @torch.inference_mode()
    def encode_vision(self, frames: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode frames once so generation can reuse a cached-vision prefix."""
        visual_device = self._get_visual_device()

        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": "."})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(visual_device, dtype=self._get_visual_dtype())
        image_grid_thw = inputs["image_grid_thw"].to(visual_device)
        image_embeds = self._flatten_vision_features(
            self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
        )
        return image_embeds, image_grid_thw

    @torch.inference_mode()
    def generate_with_cached_vision(
        self,
        cached_embeds: torch.Tensor,
        cached_grid_thw: torch.Tensor,
        question: str,
    ) -> str:
        """Generate from cached vision embeddings using a single explicit vision block."""
        text_device = self._get_text_input_device()
        tokenizer = self.processor.tokenizer
        multimodal_model = self._get_multimodal_model()

        num_vision_tokens = int(cached_embeds.shape[0])
        self._last_num_vision_tokens = num_vision_tokens
        self._last_num_vision_frames = int(cached_grid_thw.shape[0]) if cached_grid_thw is not None else 0

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        input_ids_list: list[int] = []
        input_ids_list.extend([self._im_start_id])
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        input_ids_list.append(self._vision_start_id)
        input_ids_list.extend([self.image_token_id] * num_vision_tokens)
        input_ids_list.append(self._vision_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend(question_ids)
        input_ids_list.append(self._im_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend([self._im_start_id])
        input_ids_list.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=text_device)
        attention_mask = torch.ones_like(input_ids)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        cached_embeds = cached_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, cached_embeds)

        position_ids, _ = multimodal_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=cached_grid_thw.to(inputs_embeds.device),
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return self._generate_from_model_inputs(
            prompt_length=int(input_ids.shape[1]),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    @torch.inference_mode()
    def generate_from_frames(self, frames: list[Image.Image], question: str) -> str:
        """Generate from frames via cached vision encoding and explicit prefix construction."""
        cached_embeds, cached_grid_thw = self.encode_vision(frames)
        return self.generate_with_cached_vision(cached_embeds, cached_grid_thw, question)


def build_ovo_prompt(task: str, anno: dict[str, Any], index: int = 0) -> str:
    if task in ALL_BR_TASKS:
        options = anno["options"]
        opts_str = "; ".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)) + ";"
        return BR_PROMPT_TEMPLATE.format(anno["question"], opts_str)
    if task == "REC":
        return REC_PROMPT_TEMPLATE.format(f"How many times did they {anno['activity']}?")
    if task == "SSR":
        return SSR_PROMPT_TEMPLATE.format(anno["test_info"][index]["step"])
    if task == "CRR":
        return CRR_PROMPT_TEMPLATE.format(anno["question"])
    return anno.get("question", "")


def extract_mcq_answer(response: str | None) -> str | None:
    if response is None or not str(response).strip():
        return None
    text = str(response).strip().upper()
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([1-4])\b", text)
    if match:
        return chr(64 + int(match.group(1)))
    return None


def score_ovo_br(response: str | None, gt: str) -> int:
    pred = extract_mcq_answer(response)
    return int(pred is not None and pred.upper() == gt.upper())


def score_ovo_rec(response: str | None, gt_count: int) -> int:
    if response is None or not str(response).strip():
        return 0
    nums = re.findall(r"\d+", str(response))
    return int("".join(nums) == str(gt_count)) if nums else 0


def score_yes_no(response: str | None, gt_type: int) -> int:
    if response is None or not str(response).strip():
        return 0
    text = str(response).strip().upper()
    if (text == "N" or "NO" in text) and gt_type == 0:
        return 1
    if (text == "Y" or "YES" in text) and gt_type == 1:
        return 1
    return 0


def calculate_ovo_scores(backward_results: list[dict], realtime_results: list[dict], forward_results: list[dict]) -> dict[str, Any]:
    summary: dict[str, Any] = {"backward": {}, "realtime": {}, "forward": {}}

    for section_name, results in (("backward", backward_results), ("realtime", realtime_results)):
        by_task: dict[str, list[int]] = defaultdict(list)
        for result in results:
            by_task[result["task"]].append(score_ovo_br(result.get("response"), result["ground_truth"]))
        for task, vals in by_task.items():
            summary[section_name][task] = {
                "correct": sum(vals),
                "total": len(vals),
                "accuracy": 100.0 * sum(vals) / len(vals),
            }

    by_task: dict[str, list[int]] = defaultdict(list)
    for result in forward_results:
        task = result["task"]
        if task == "REC":
            for item in result["test_info"]:
                by_task["REC"].append(score_ovo_rec(item.get("response"), item["count"]))
        elif task in {"SSR", "CRR"}:
            for item in result["test_info"]:
                by_task[task].append(score_yes_no(item.get("response"), item["type"]))
    for task, vals in by_task.items():
        summary["forward"][task] = {
            "correct": sum(vals),
            "total": len(vals),
            "accuracy": 100.0 * sum(vals) / len(vals),
        }
    return summary


def print_ovo_results(model_label: str, backward_results: list[dict], realtime_results: list[dict], forward_results: list[dict]) -> None:
    summary = calculate_ovo_scores(backward_results, realtime_results, forward_results)
    print("\n" + "=" * 60)
    print(f"OVO-Bench Recent-Window Results ({model_label})")
    print("=" * 60)

    category_scores: list[float] = []
    for section_name, title in (
        ("backward", "Backward Tracing"),
        ("realtime", "Real-time Perception"),
        ("forward", "Forward Responding"),
    ):
        rows = summary[section_name]
        if not rows:
            continue
        print(f"\n{title}:")
        accs: list[float] = []
        for task, stats in rows.items():
            print(f"  {task}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
            accs.append(float(stats["accuracy"]))
        avg = sum(accs) / len(accs)
        category_scores.append(avg)
        print(f"  {title.split()[0]} Avg.: {avg:.2f}%")

    if category_scores:
        total_avg = sum(category_scores) / len(category_scores)
        print(f"\n{'=' * 60}")
        print(f"Total Avg.: {total_avg:.2f}%")
        print("=" * 60)


def decode_video_to_chunks_qwen(
    video_path: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int | None = None,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[list[EvalChunk], str]:
    use_exact_recent = os.environ.get("QWEN_EXACT_RECENT_DECODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_exact_recent:
        try:
            from lib.qwen_exact_recent_decoder import fetch_recent_video_exact
        except ImportError as exc:
            raise RuntimeError("Exact recent decoder is required when QWEN_EXACT_RECENT_DECODE=1.") from exc
    else:
        try:
            from qwen_vl_utils.vision_process import fetch_video
        except ImportError as exc:
            raise RuntimeError("qwen_vl_utils is required for video decoding.") from exc

    if chunk_duration <= 0:
        raise ValueError(f"chunk_duration must be > 0, got {chunk_duration}")

    video_req: dict[str, Any] = {"video": video_path, "fps": float(fps)}
    if video_start is not None:
        video_req["video_start"] = max(0.0, float(video_start))
    if video_end is not None:
        video_req["video_end"] = max(0.0, float(video_end))

    if use_exact_recent:
        if recent_frames_only is None or int(recent_frames_only) < 1:
            raise ValueError("recent_frames_only must be >= 1 when QWEN_EXACT_RECENT_DECODE=1")
        if abs(float(chunk_duration) * float(fps) - 1.0) > 1e-6:
            raise ValueError(
                "QWEN_EXACT_RECENT_DECODE currently requires chunk_duration * fps == 1.0 "
                "so that the last N decoded frames match the last N recent-window chunks exactly."
            )
        video, metadata = fetch_recent_video_exact(
            video_req,
            last_nframes=int(recent_frames_only),
            return_video_metadata=True,
        )
    else:
        video, metadata = fetch_video(video_req, return_video_metadata=True)

    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError(f"Unexpected qwen_vl_utils output for video={video_path!r}")

    meta = metadata if isinstance(metadata, dict) else {}
    raw_fps = max(float(meta.get("fps", fps if fps > 0 else 1.0)), 1e-6)
    frame_indices = meta.get("frames_indices")
    if isinstance(frame_indices, torch.Tensor):
        frame_indices = frame_indices.detach().cpu().reshape(-1).tolist()
    elif frame_indices is not None and not isinstance(frame_indices, (list, tuple)):
        try:
            frame_indices = list(frame_indices)
        except TypeError:
            frame_indices = None
    if frame_indices is None or len(frame_indices) != int(video.shape[0]):
        start_frame = int(max(0.0, float(video_start or 0.0)) * raw_fps)
        frame_indices = [start_frame + i for i in range(int(video.shape[0]))]
    frame_indices = [int(x) for x in frame_indices]

    if len(frame_indices) > 1:
        sampled_duration = float(frame_indices[-1] - frame_indices[0]) / raw_fps
        sampled_fps = float(len(frame_indices) - 1) / max(sampled_duration, 1e-6)
    else:
        sampled_fps = max(float(fps), 1e-6)
    decode_backend = str(meta.get("video_backend", "unknown"))
    if video_start is not None or video_end is not None:
        decode_backend = f"{decode_backend}_window"

    max_ts = max((float(idx) / raw_fps for idx in frame_indices), default=0.0)
    if len(frame_indices) > 1:
        frame_dt = max(float(frame_indices[-1] - frame_indices[-2]) / raw_fps, 1.0 / raw_fps)
    else:
        frame_dt = 1.0 / raw_fps
    max_valid_end = max_ts + frame_dt

    frame_buckets: dict[int, list[tuple[Image.Image, float]]] = {}
    for i, frame_idx in enumerate(frame_indices):
        ts = float(frame_idx) / raw_fps
        chunk_idx = int(ts // chunk_duration)
        frame = video[i].clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        frame_buckets.setdefault(chunk_idx, []).append((Image.fromarray(frame), ts))
    del video

    chunks: list[EvalChunk] = []
    for chunk_idx in sorted(frame_buckets):
        chunk_frames = frame_buckets[chunk_idx]
        chunks.append(
            EvalChunk(
                frames=[frame for frame, _ in chunk_frames],
                frame_timestamps=[ts for _, ts in chunk_frames],
                start_time=chunk_idx * chunk_duration,
                end_time=min((chunk_idx + 1) * chunk_duration, max_valid_end),
                chunk_index=chunk_idx,
                fps=sampled_fps,
            )
        )
    return chunks, decode_backend


def query_recent_window(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str]:
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames_only,
        video_start=video_start,
        video_end=video_end,
    )
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    window_size = max(1, int(recent_frames_only))
    recent_chunks = list(chunks[-window_size:])
    final_frames: list[Image.Image] = []
    for chunk in recent_chunks:
        final_frames.extend(chunk.frames)

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(final_frames, prompt)
    final_chunk_ids = [item.chunk_index for item in recent_chunks]

    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = qa._last_num_vision_tokens
    num_frames = qa._last_num_vision_frames

    return (
        RecentWindowResult(
            answer=answer,
            final_chunk_ids=final_chunk_ids,
            generate_time=generate_time,
            ttft_seconds=ttft_seconds,
            num_vision_tokens=num_vision_tokens,
            num_vision_tokens_before=num_vision_tokens,
            num_vision_tokens_after=num_vision_tokens,
            num_frames=num_frames,
        ),
        decode_backend,
    )


def evaluate_ovo_backward_realtime(
    anno: dict[str, Any],
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
) -> dict[str, Any]:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    response = None
    metadata: dict[str, Any] = {}
    if os.path.exists(video_path):
        result, decode_backend = query_recent_window(
            qa=qa,
            video_path=video_path,
            prompt=build_ovo_prompt(anno["task"], anno),
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
        )
        response = result.answer
        metadata = {
            "decode_backend": decode_backend,
            "final_chunk_ids": result.final_chunk_ids,
            "generate_time": result.generate_time,
            "ttft_seconds": result.ttft_seconds,
            "num_vision_tokens": result.num_vision_tokens,
            "num_vision_tokens_before": result.num_vision_tokens_before,
            "num_vision_tokens_after": result.num_vision_tokens_after,
            "num_frames": result.num_frames,
        }
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "response": response,
        "ground_truth": chr(65 + anno["gt"]),
        **metadata,
    }


def evaluate_ovo_forward(
    anno: dict[str, Any],
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
) -> dict[str, Any]:
    result_anno = copy.deepcopy(anno)
    for index, test_info in enumerate(result_anno["test_info"]):
        video_path = os.path.join(chunked_dir, f"{anno['id']}_{index}.mp4")
        if not os.path.exists(video_path):
            test_info["response"] = None
            continue
        result, decode_backend = query_recent_window(
            qa=qa,
            video_path=video_path,
            prompt=build_ovo_prompt(anno["task"], anno, index=index),
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
        )
        test_info["response"] = result.answer
        test_info["decode_backend"] = decode_backend
        test_info["final_chunk_ids"] = result.final_chunk_ids
        test_info["generate_time"] = result.generate_time
        test_info["ttft_seconds"] = result.ttft_seconds
    return result_anno


def flatten_gathered_results(gathered: list[Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for item in gathered:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def load_jsonl_results(path: str) -> tuple[list[dict[str, Any]], set[str]]:
    results: list[dict[str, Any]] = []
    done_keys: set[str] = set()
    if not os.path.exists(path):
        return results, done_keys
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            results.append(item)
            key = item.get("_key")
            if isinstance(key, str) and key:
                done_keys.add(key)
    return results, done_keys


def save_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
