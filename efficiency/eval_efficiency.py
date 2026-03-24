#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import av
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = SCRIPT_DIR / "video"
DEFAULT_RESULT_ROOT = SCRIPT_DIR / "result"
DEFAULT_FRAME_COUNTS = (16, 64, 256, 512)
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROMPT = "Describe the video in detail."


@dataclass
class BenchmarkRow:
    total_frames: int
    chunk_size: int
    recent_frames: int
    model_input_frames: int
    num_chunks: int
    generated_tokens: int
    vision_encode_s: float
    ttft_s: float
    e2e_ttft_s: float
    tpot_s: float | None
    total_generate_s: float
    model_compute_total_s: float
    end_to_end_total_s: float
    history_overhead_s: float
    decode_tokens_per_s: float
    end_to_end_tokens_per_s: float
    model_latency_per_input_frame_ms: float
    end_to_end_latency_per_output_token_ms: float
    start_allocated_gb: float
    peak_memory_gb: float
    delta_peak_memory_gb: float
    response: str
    video_path: str


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary `{name}` is not available in PATH.")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _probe_frame_count(video_path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        text=True,
    ).strip()
    return int(out)


def _prepare_sample_video(
    source_video: Path,
    output_dir: Path,
    frame_count: int,
    side_pixels: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_video.stem}_{frame_count:04d}f_{side_pixels}px.mp4"
    if output_path.exists():
        try:
            if _probe_frame_count(output_path) == frame_count:
                return output_path
        except Exception:
            pass

    vf = f"scale={side_pixels}:{side_pixels}:flags=lanczos"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_video),
            "-an",
            "-vf",
            vf,
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    actual = _probe_frame_count(output_path)
    if actual != frame_count:
        raise RuntimeError(f"{output_path} frame_count mismatch: expected {frame_count}, got {actual}")
    return output_path


def _load_all_frames(video_path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
    return frames


def _recent_frames_from_stream(
    frames: list[Image.Image],
    chunk_size: int,
    recent_frames: int,
) -> tuple[list[Image.Image], int]:
    chunks = [frames[i : i + chunk_size] for i in range(0, len(frames), chunk_size)]
    streamed: list[Image.Image] = []
    for chunk in chunks:
        streamed.extend(chunk)
    return streamed[-recent_frames:], len(chunks)


def _masked_scatter_features(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    token_id: int,
    features: torch.Tensor,
    feature_name: str,
) -> torch.Tensor:
    n_tokens = (input_ids == token_id).sum().item()
    n_features = features.shape[0]
    if n_tokens != n_features:
        raise ValueError(
            f"{feature_name.capitalize()} features and tokens do not match: "
            f"tokens={n_tokens}, features={n_features}"
        )

    mask = input_ids == token_id
    mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    features = features.to(inputs_embeds.device, inputs_embeds.dtype)
    return inputs_embeds.masked_scatter(mask, features)


def _ensure_feature_tensor(features: Any, feature_name: str) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if isinstance(features, (tuple, list)):
        for item in features:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(
        f"Unexpected {feature_name} feature type: {type(features)}. "
        "Expected Tensor or tuple/list containing Tensor."
    )


def _prepare_prefill_inputs(
    model: AutoModelForImageTextToText,
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    multimodal_model = getattr(model, "model", None)
    if multimodal_model is None or not hasattr(multimodal_model, "get_rope_index"):
        raise TypeError("Internal-only TTFT measurement currently requires a Qwen2.5-VL style model.")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")
    second_per_grid_ts = inputs.get("second_per_grid_ts")

    inputs_embeds = model.get_input_embeddings()(input_ids)

    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        image_embeds = _ensure_feature_tensor(
            multimodal_model.get_image_features(pixel_values, image_grid_thw),
            feature_name="image",
        )
        inputs_embeds = _masked_scatter_features(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            token_id=model.config.image_token_id,
            features=image_embeds,
            feature_name="image",
        )

    pixel_values_videos = inputs.get("pixel_values_videos")
    if pixel_values_videos is not None:
        video_embeds = _ensure_feature_tensor(
            multimodal_model.get_video_features(pixel_values_videos, video_grid_thw),
            feature_name="video",
        )
        inputs_embeds = _masked_scatter_features(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            token_id=model.config.video_token_id,
            features=video_embeds,
            feature_name="video",
        )

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    position_ids, rope_deltas = multimodal_model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
    )
    return {
        "attention_mask": attention_mask,
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "rope_deltas": rope_deltas,
    }


@torch.inference_mode()
def _run_one(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    video_path: Path,
    prompt: str,
    chunk_size: int,
    recent_frames: int,
    max_new_tokens: int,
) -> BenchmarkRow:
    full_run_start = time.perf_counter()
    frames = _load_all_frames(video_path)
    selected_frames, num_chunks = _recent_frames_from_stream(frames, chunk_size, recent_frames)
    if not selected_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    content = [{"type": "image", "image": frame} for frame in selected_frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_allocated = torch.cuda.memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()

    model.model.rope_deltas = None
    vision_t0 = time.perf_counter()
    prefill_inputs = _prepare_prefill_inputs(model=model, inputs=inputs)
    torch.cuda.synchronize()
    vision_ready_time = time.perf_counter()

    model.model.rope_deltas = prefill_inputs["rope_deltas"]

    t0 = time.perf_counter()
    outputs = model(
        input_ids=None,
        attention_mask=prefill_inputs["attention_mask"],
        position_ids=prefill_inputs["position_ids"],
        inputs_embeds=prefill_inputs["inputs_embeds"],
        use_cache=True,
        return_dict=True,
    )
    first_token = outputs.logits[:, -1, :].argmax(dim=-1)
    torch.cuda.synchronize()
    first_token_time = time.perf_counter()

    generated_token_ids = [int(first_token.item())]
    past_key_values = outputs.past_key_values
    next_token = first_token

    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token[:, None],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated_token_ids.append(int(next_token.item()))

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    generated_tokens = len(generated_token_ids)
    response = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

    ttft_s = first_token_time - t0
    e2e_ttft_s = first_token_time - full_run_start
    tpot_s = None
    if generated_tokens > 1:
        tpot_s = (t2 - first_token_time) / float(generated_tokens - 1)

    model_compute_total_s = (vision_ready_time - vision_t0) + (t2 - t0)
    end_to_end_total_s = t2 - full_run_start
    history_overhead_s = max(0.0, end_to_end_total_s - model_compute_total_s)
    decode_tokens_per_s = generated_tokens / max(t2 - t0, 1e-9)
    end_to_end_tokens_per_s = generated_tokens / max(end_to_end_total_s, 1e-9)
    model_latency_per_input_frame_ms = (model_compute_total_s * 1000.0) / max(len(selected_frames), 1)
    end_to_end_latency_per_output_token_ms = (end_to_end_total_s * 1000.0) / max(generated_tokens, 1)

    return BenchmarkRow(
        total_frames=len(frames),
        chunk_size=chunk_size,
        recent_frames=recent_frames,
        model_input_frames=len(selected_frames),
        num_chunks=num_chunks,
        generated_tokens=generated_tokens,
        vision_encode_s=vision_ready_time - vision_t0,
        ttft_s=ttft_s,
        e2e_ttft_s=e2e_ttft_s,
        tpot_s=tpot_s,
        total_generate_s=t2 - t0,
        model_compute_total_s=model_compute_total_s,
        end_to_end_total_s=end_to_end_total_s,
        history_overhead_s=history_overhead_s,
        decode_tokens_per_s=decode_tokens_per_s,
        end_to_end_tokens_per_s=end_to_end_tokens_per_s,
        model_latency_per_input_frame_ms=model_latency_per_input_frame_ms,
        end_to_end_latency_per_output_token_ms=end_to_end_latency_per_output_token_ms,
        start_allocated_gb=start_allocated,
        peak_memory_gb=torch.cuda.max_memory_allocated() / (1024**3),
        delta_peak_memory_gb=(torch.cuda.max_memory_allocated() / (1024**3)) - start_allocated,
        response=response,
        video_path=str(video_path),
    )


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _build_run_dir(result_root: Path, model_name: str, chunk_size: int, recent_frames: int) -> Path:
    model_slug = _slugify(model_name.split("/")[-1])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = result_root / f"{model_slug}_chunk{chunk_size}_recent{recent_frames}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _detect_runtime_env() -> str:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix).name

    python_path = Path(sys.executable).resolve()
    parts = python_path.parts
    if "envs" in parts:
        idx = parts.index("envs")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    return os.environ.get("CONDA_DEFAULT_ENV", "")


def _write_results(rows: list[BenchmarkRow], output_dir: Path, meta: dict[str, Any]) -> None:
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    md_path = output_dir / "summary.md"

    payload = {
        "meta": meta,
        "results": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    lines = [
        f"# {meta['model_name']} Efficiency Result",
        "",
        f"- Model: `{meta['model_name']}`",
        f"- Source Video: `{meta['source_video']}`",
        f"- Source Video SHA256: `{meta['source_video_sha256']}`",
        f"- Python: `{meta['python_executable']}`",
        f"- Conda Env: `{meta['conda_env']}`",
        f"- Torch: `{meta['torch_version']}`",
        f"- Transformers: `{meta['transformers_version']}`",
        f"- Device: `{meta['device_name']}`",
        f"- Chunk Size: `{meta['chunk_size']}`",
        f"- Recent Frames: `{meta['recent_frames']}`",
        f"- Prompt: `{meta['prompt']}`",
        f"- Max New Tokens: `{meta['max_new_tokens']}`",
        f"- Attention Implementation: `{meta['attn_implementation']}`",
        f"- TTFT Boundary: `{meta['ttft_boundary']}`",
        f"- E2E TTFT Boundary: `{meta['e2e_ttft_boundary']}`",
        f"- E2E Total Boundary: `{meta['e2e_total_boundary']}`",
        f"- Fair Metric Note: `{meta['fair_metric_note']}`",
        "",
        "| Frames | Chunks | InFrames | GenTok | Vision (s) | TTFT-internal (s) | TTFT-E2E (s) | TPOP (s) | Model-E2E (s) | End2End (s) | Tok/s E2E | Peak Mem (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        tpop_str = f"{row.tpot_s:.3f}" if row.tpot_s is not None else "-"
        lines.append(
            f"| {row.total_frames} | {row.num_chunks} | {row.model_input_frames} | {row.generated_tokens} | "
            f"{row.vision_encode_s:.3f} | {row.ttft_s:.3f} | {row.e2e_ttft_s:.3f} | {tpop_str} | "
            f"{row.model_compute_total_s:.3f} | {row.end_to_end_total_s:.3f} | "
            f"{row.end_to_end_tokens_per_s:.2f} | {row.peak_memory_gb:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-video",
        type=Path,
        required=True,
        help="Path to the long source video used to generate the 16/64/256/512-frame benchmark clips.",
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name or local path.",
    )
    parser.add_argument("--frame-counts", type=int, nargs="+", default=list(DEFAULT_FRAME_COUNTS))
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--recent-frames", type=int, default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--video-side-pixels", type=int, default=128)
    parser.add_argument("--max-pixels", type=int, default=128 * 128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_binary("ffmpeg")
    _require_binary("ffprobe")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not args.source_video.exists():
        raise FileNotFoundError(f"Source video not found: {args.source_video}")

    sample_videos = [
        _prepare_sample_video(
            source_video=args.source_video,
            output_dir=args.video_dir,
            frame_count=frame_count,
            side_pixels=args.video_side_pixels,
        )
        for frame_count in args.frame_counts
    ]

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        max_pixels=args.max_pixels,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.to("cuda")
    model.eval()

    _run_one(
        model=model,
        processor=processor,
        video_path=sample_videos[0],
        prompt=args.prompt,
        chunk_size=args.chunk_size,
        recent_frames=args.recent_frames,
        max_new_tokens=min(8, args.max_new_tokens),
    )
    torch.cuda.empty_cache()

    rows: list[BenchmarkRow] = []
    for video_path in sample_videos:
        row = _run_one(
            model=model,
            processor=processor,
            video_path=video_path,
            prompt=args.prompt,
            chunk_size=args.chunk_size,
            recent_frames=args.recent_frames,
            max_new_tokens=args.max_new_tokens,
        )
        rows.append(row)
        tpop_str = f"{row.tpot_s:.3f}s" if row.tpot_s is not None else "-"
        print(
            f"{row.total_frames:>4}f  VisionEncode={row.vision_encode_s:.3f}s  "
            f"TTFT(internal)={row.ttft_s:.3f}s  "
            f"TTFT(E2E)={row.e2e_ttft_s:.3f}s  "
            f"TPOP={tpop_str}  "
            f"E2E={row.end_to_end_total_s:.3f}s  "
            f"Tok/s(E2E)={row.end_to_end_tokens_per_s:.2f}  "
            f"Peak={row.peak_memory_gb:.3f}GB"
        )

    meta = {
        "source_video": str(args.source_video),
        "source_video_sha256": _sha256_file(args.source_video),
        "model_name": args.model_name,
        "python_executable": sys.executable,
        "conda_env": _detect_runtime_env(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "chunk_size": args.chunk_size,
        "recent_frames": args.recent_frames,
        "prompt": args.prompt,
        "max_pixels": args.max_pixels,
        "video_side_pixels": args.video_side_pixels,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "greedy_decoding": True,
        "fixed_decode_length": True,
        "warmup_run": True,
        "ttft_boundary": "after multimodal embeddings and RoPE indices are ready; vision encoding excluded",
        "e2e_ttft_boundary": "from entering _run_one() (frame decode/select + processor + vision + decode) to first generated token",
        "e2e_total_boundary": "from entering _run_one() to completion of fixed-length generation",
        "fair_metric_note": "For comparisons with KV-cache methods, prioritize end_to_end_total_s, e2e_ttft_s, and end_to_end_tokens_per_s over decode-only TTFT.",
    }
    run_dir = _build_run_dir(
        result_root=args.result_root,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        recent_frames=args.recent_frames,
    )
    _write_results(rows, run_dir, meta)
    print(f"archived_results={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
