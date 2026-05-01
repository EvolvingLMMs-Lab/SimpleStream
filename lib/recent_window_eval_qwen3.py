from __future__ import annotations

import copy
import os
import time
from collections import deque
from dataclasses import dataclass

import torch
from PIL import Image

from lib.recent_window_eval import (
    RecentWindowQAModel as _BaseRecentWindowQAModel,
    RecentWindowResult,
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
    flatten_gathered_results,
    print_ovo_results,
)


class RecentWindowQAModel(_BaseRecentWindowQAModel):
    """Qwen3 release wrapper aligned with the per-frame vision-token builder."""

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self._last_ttft_seconds = 0.0
        self._last_num_vision_tokens = 0
        self._last_num_vision_frames = 0

        proc_kwargs: dict[str, object] = {}
        if os.environ.get("MIN_PIXELS"):
            proc_kwargs["min_pixels"] = int(os.environ["MIN_PIXELS"])
        if os.environ.get("MAX_PIXELS"):
            proc_kwargs["max_pixels"] = int(os.environ["MAX_PIXELS"])
        self.processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)

        model_kwargs: dict[str, object] = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"

        saved_world_size = os.environ.pop("WORLD_SIZE", None)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        finally:
            if saved_world_size is not None:
                os.environ["WORLD_SIZE"] = saved_world_size
        if device != "auto":
            self.model.to(device)
        self.model.eval()

        self._hf_model = self.model
        self._visual = self.model.model.visual
        self._text_model = self.model.model
        self.image_token_id = self.model.config.image_token_id
        self.vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.merge_size = self.model.model.visual.spatial_merge_size

    @torch.inference_mode()
    def encode_vision(self, frames: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep official preprocessing, but expose encoded vision for explicit input building."""
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

        pixel_values = inputs["pixel_values"].to(self.model.device)
        image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
        image_embeds = self._flatten_vision_features(
            self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
        )

        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return image_embeds, image_grid_thw

    @torch.inference_mode()
    def encode_vision_batched(
        self,
        frames_per_chunk: list[list[Image.Image]],
        max_frames_per_batch: int = 8,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not frames_per_chunk:
            return []

        flat_pairs: list[tuple[int, Image.Image]] = []
        for chunk_index, frames in enumerate(frames_per_chunk):
            for frame in frames:
                flat_pairs.append((chunk_index, frame))

        hidden_size = int(getattr(self.model.config, "hidden_size", 4096))
        model_dtype = getattr(self.model, "dtype", torch.bfloat16)
        empty_emb = torch.empty((0, hidden_size), dtype=model_dtype, device="cpu")
        empty_grid = torch.empty((0, 3), dtype=torch.long, device="cpu")
        if not flat_pairs:
            return [(empty_emb, empty_grid) for _ in frames_per_chunk]

        merge_area = max(1, int(self.merge_size)) ** 2
        chunk_embeds: list[list[torch.Tensor]] = [[] for _ in frames_per_chunk]
        chunk_grids: list[list[torch.Tensor]] = [[] for _ in frames_per_chunk]

        batch_size = max(1, int(max_frames_per_batch))
        offset_flat = 0
        while offset_flat < len(flat_pairs):
            pairs = flat_pairs[offset_flat : offset_flat + batch_size]
            content = [{"type": "image", "image": frame} for _, frame in pairs]
            content.append({"type": "text", "text": "."})
            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self.model.device)
            image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
            image_embeds = self._flatten_vision_features(
                self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
            )

            frame_token_counts = [
                max(1, int(row[0].item() * row[1].item() * row[2].item()) // merge_area)
                for row in image_grid_thw
            ]
            expected_tokens = sum(frame_token_counts)
            if expected_tokens != int(image_embeds.shape[0]) or len(frame_token_counts) != len(pairs):
                grouped: dict[int, list[Image.Image]] = {}
                for chunk_index, frame in pairs:
                    grouped.setdefault(chunk_index, []).append(frame)
                for chunk_index, frames in grouped.items():
                    emb, grid = self.encode_vision(frames)
                    chunk_embeds[chunk_index].append(emb.to(dtype=torch.bfloat16, device="cpu"))
                    chunk_grids[chunk_index].append(grid.cpu())
                offset_flat += len(pairs)
                del pixel_values, image_grid_thw, image_embeds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            offset = 0
            for (chunk_index, _), token_count, row in zip(pairs, frame_token_counts, image_grid_thw):
                end = offset + token_count
                chunk_embeds[chunk_index].append(image_embeds[offset:end].to(dtype=torch.bfloat16, device="cpu"))
                chunk_grids[chunk_index].append(row.unsqueeze(0).cpu())
                offset = end
            offset_flat += len(pairs)

            del pixel_values, image_grid_thw, image_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for chunk_index in range(len(frames_per_chunk)):
            if chunk_embeds[chunk_index]:
                outputs.append(
                    (
                        torch.cat(chunk_embeds[chunk_index], dim=0),
                        torch.cat(chunk_grids[chunk_index], dim=0),
                    )
                )
            else:
                outputs.append((empty_emb, empty_grid))
        return outputs

    @torch.inference_mode()
    def generate_with_vision_features(
        self,
        vision_embeds: torch.Tensor,
        vision_grid_thw: torch.Tensor,
        question: str,
    ) -> str:
        device = self.model.device
        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model()

        num_vision_tokens = int(vision_embeds.shape[0])
        self._last_num_vision_tokens = num_vision_tokens
        self._last_num_vision_frames = int(vision_grid_thw.shape[0]) if vision_grid_thw is not None else 0

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        grid_rows = vision_grid_thw.to(device)
        tokens_per_frame = (grid_rows.prod(dim=-1) // (self.merge_size**2)).tolist()
        expected_tokens = sum(int(n) for n in tokens_per_frame)
        if expected_tokens != num_vision_tokens:
            raise ValueError(
                "vision token count mismatch: "
                f"embeds={num_vision_tokens} vs grid={expected_tokens}"
            )

        input_ids_list: list[int] = []
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        for frame_token_count in tokens_per_frame:
            input_ids_list.append(self.vision_start_id)
            input_ids_list.extend([self.image_token_id] * int(frame_token_count))
            input_ids_list.append(self.vision_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend(question_ids)
        input_ids_list.append(self.im_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, vision_embeds)

        position_ids, _ = text_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=grid_rows,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        return self._generate_from_model_inputs(
            prompt_length=len(input_ids[0]),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    @torch.inference_mode()
    def generate_from_frames(self, frames: list[Image.Image], question: str) -> str:
        vision_embeds, vision_grid_thw = self.encode_vision(frames)
        return self.generate_with_vision_features(vision_embeds, vision_grid_thw, question)


@dataclass
class EncodedChunk:
    vision_emb: torch.Tensor
    grid_thw: torch.Tensor
    chunk_index: int
    start_time: float
    end_time: float


def _combine_window_embeddings(
    window: deque[EncodedChunk],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_embeds = torch.cat([item.vision_emb.to(device) for item in window], dim=0)
    combined_grid_thw = torch.cat([item.grid_thw.to(device) for item in window], dim=0)
    return combined_embeds, combined_grid_thw


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
    recent_chunks = chunks[-window_size:]
    encoded_chunks: list[EncodedChunk] = []
    encoded_outputs = qa.encode_vision_batched([chunk.frames for chunk in recent_chunks], max_frames_per_batch=8)
    for chunk, (vision_emb, grid_thw) in zip(recent_chunks, encoded_outputs):
        if int(vision_emb.shape[0]) == 0 or int(grid_thw.shape[0]) == 0:
            continue
        encoded_chunks.append(
            EncodedChunk(
                vision_emb=vision_emb,
                grid_thw=grid_thw,
                chunk_index=chunk.chunk_index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
        )
    if not encoded_chunks:
        raise ValueError(f"No vision chunks encoded from video: {video_path}")

    encoded_window: deque[EncodedChunk] = deque(encoded_chunks, maxlen=window_size)
    t0 = time.perf_counter()
    combined_embeds, combined_grid_thw = _combine_window_embeddings(encoded_window, qa.model.device)
    answer = qa.generate_with_vision_features(combined_embeds, combined_grid_thw, prompt)
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = qa._last_num_vision_tokens
    num_frames = qa._last_num_vision_frames

    return (
        RecentWindowResult(
            answer=answer,
            final_chunk_ids=[item.chunk_index for item in encoded_window],
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
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
) -> dict:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    response = None
    metadata: dict = {}
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
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
) -> dict:
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
        test_info["num_vision_tokens"] = result.num_vision_tokens
        test_info["num_vision_tokens_before"] = result.num_vision_tokens_before
        test_info["num_vision_tokens_after"] = result.num_vision_tokens_after
        test_info["num_frames"] = result.num_frames
    return result_anno
