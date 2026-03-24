from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from qwen_vl_utils.vision_process import (
    FRAME_FACTOR,
    FPS,
    MODEL_SEQ_LEN,
    SPATIAL_MERGE_SIZE,
    VIDEO_MAX_TOKEN_NUM,
    VIDEO_MIN_TOKEN_NUM,
    calculate_video_frame_range,
    get_video_reader_backend,
    smart_nframes,
    smart_resize,
)


@dataclass
class ExactRecentSamplingPlan:
    backend: str
    raw_fps: float
    raw_total_frames: int
    start_frame: int
    end_frame: int
    available_frames: int
    sampled_nframes_full: int
    sample_fps: float
    full_indices: list[int]
    tail_indices: list[int]


def _build_sampling_plan(ele: dict[str, Any], last_nframes: int) -> ExactRecentSamplingPlan:
    if not isinstance(ele.get("video"), str):
        raise TypeError("Exact recent decoding currently supports path-like video inputs only.")

    backend = get_video_reader_backend()
    video_path = ele["video"]

    if backend == "decord":
        import decord

        reader = decord.VideoReader(video_path)
        raw_total_frames = len(reader)
        raw_fps = float(reader.get_avg_fps())
    elif backend == "torchcodec":
        import os

        from torchcodec.decoders import VideoDecoder

        num_threads = int(os.environ.get("TORCHCODEC_NUM_THREADS", 8))
        reader = VideoDecoder(video_path, num_ffmpeg_threads=num_threads)
        raw_total_frames = int(reader.metadata.num_frames)
        raw_fps = float(reader.metadata.average_fps)
    else:
        raise NotImplementedError(
            f"Exact recent decoding is implemented for decord/torchcodec only, got backend={backend!r}."
        )

    start_frame, end_frame, available_frames = calculate_video_frame_range(ele, raw_total_frames, raw_fps)
    sampled_nframes_full = smart_nframes(ele, total_frames=available_frames, video_fps=raw_fps)
    full_indices = torch.linspace(start_frame, end_frame, sampled_nframes_full).round().long().tolist()
    tail_count = max(1, int(last_nframes))
    tail_indices = full_indices[-tail_count:]
    sample_fps = sampled_nframes_full / max(float(available_frames), 1e-6) * raw_fps

    return ExactRecentSamplingPlan(
        backend=backend,
        raw_fps=raw_fps,
        raw_total_frames=raw_total_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        available_frames=available_frames,
        sampled_nframes_full=sampled_nframes_full,
        sample_fps=sample_fps,
        full_indices=full_indices,
        tail_indices=tail_indices,
    )


def _decode_indices(video_path: str, backend: str, indices: list[int]) -> torch.Tensor:
    if backend == "decord":
        import decord

        reader = decord.VideoReader(video_path)
        video = reader.get_batch(indices).asnumpy()
        return torch.tensor(video).permute(0, 3, 1, 2)

    if backend == "torchcodec":
        import os

        from torchcodec.decoders import VideoDecoder

        num_threads = int(os.environ.get("TORCHCODEC_NUM_THREADS", 8))
        reader = VideoDecoder(video_path, num_ffmpeg_threads=num_threads)
        return reader.get_frames_at(indices=indices).data

    raise NotImplementedError(f"Unsupported backend: {backend!r}")


def _resize_like_fetch_video(
    video: torch.Tensor,
    ele: dict[str, Any],
    sampled_nframes_full: int,
    image_patch_size: int,
) -> torch.Tensor:
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    video_frame_min_pixels = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    video_frame_max_pixels = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor

    _, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", video_frame_min_pixels)
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
    max_pixels = max(
        min(video_frame_max_pixels, total_pixels / sampled_nframes_full * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels = min(ele.get("max_pixels", max_pixels), max_pixels)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    return transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()


def fetch_recent_video_exact(
    ele: dict[str, Any],
    last_nframes: int,
    image_patch_size: int = 14,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> Any:
    plan = _build_sampling_plan(ele, last_nframes=last_nframes)
    video = _decode_indices(ele["video"], plan.backend, plan.tail_indices)
    video = _resize_like_fetch_video(
        video=video,
        ele=ele,
        sampled_nframes_full=plan.sampled_nframes_full,
        image_patch_size=image_patch_size,
    )

    video_metadata = dict(
        fps=plan.raw_fps,
        frames_indices=plan.tail_indices,
        total_num_frames=plan.available_frames,
        video_backend=f"{plan.backend}_exact_recent",
        full_sampled_nframes=plan.sampled_nframes_full,
        full_sampled_indices=plan.full_indices,
        start_frame=plan.start_frame,
        end_frame=plan.end_frame,
    )

    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, plan.sample_fps
    return final_video


def verify_recent_video_exact_matches_full(
    ele: dict[str, Any],
    last_nframes: int,
    image_patch_size: int = 14,
) -> dict[str, Any]:
    from qwen_vl_utils.vision_process import fetch_video

    full_video, full_metadata = fetch_video(ele, image_patch_size=image_patch_size, return_video_metadata=True)
    exact_video, exact_metadata = fetch_recent_video_exact(
        ele,
        last_nframes=last_nframes,
        image_patch_size=image_patch_size,
        return_video_metadata=True,
    )

    expected_video = full_video[-len(exact_metadata["frames_indices"]) :]
    expected_indices = list(full_metadata["frames_indices"][-len(exact_metadata["frames_indices"]) :])

    same_shape = tuple(expected_video.shape) == tuple(exact_video.shape)
    same_indices = expected_indices == list(exact_metadata["frames_indices"])
    same_pixels = bool(torch.equal(expected_video, exact_video))
    max_abs_diff = float((expected_video - exact_video).abs().max().item()) if same_shape else float("inf")

    return {
        "backend_full": full_metadata.get("video_backend"),
        "backend_exact": exact_metadata.get("video_backend"),
        "expected_indices": expected_indices,
        "actual_indices": list(exact_metadata["frames_indices"]),
        "same_indices": same_indices,
        "same_shape": same_shape,
        "same_pixels": same_pixels,
        "max_abs_diff": max_abs_diff,
        "expected_shape": tuple(expected_video.shape),
        "actual_shape": tuple(exact_video.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact tail decoder matching qwen_vl_utils full sampling.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--last-nframes", type=int, required=True)
    parser.add_argument("--video-start", type=float, default=None)
    parser.add_argument("--video-end", type=float, default=None)
    parser.add_argument("--verify-full", action="store_true")
    args = parser.parse_args()

    ele: dict[str, Any] = {"video": args.video, "fps": args.fps}
    if args.video_start is not None:
        ele["video_start"] = args.video_start
    if args.video_end is not None:
        ele["video_end"] = args.video_end

    plan = _build_sampling_plan(ele, last_nframes=args.last_nframes)
    print(f"backend={plan.backend}")
    print(f"raw_fps={plan.raw_fps}")
    print(f"raw_total_frames={plan.raw_total_frames}")
    print(f"sampled_nframes_full={plan.sampled_nframes_full}")
    print(f"tail_indices={plan.tail_indices}")

    if args.verify_full:
        result = verify_recent_video_exact_matches_full(ele, last_nframes=args.last_nframes)
        for key, value in result.items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
