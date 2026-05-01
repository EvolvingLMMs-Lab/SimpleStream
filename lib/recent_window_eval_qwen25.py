from __future__ import annotations

import copy
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from lib.recent_window_eval import (
    RecentWindowQAModel as _BaseRecentWindowQAModel,
    RecentWindowResult,
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
    print_ovo_results,
)
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS

QWEN25_BR_PROMPT_TEMPLATE = (
    "{question}\n"
    "Options: {options}\n"
    "Only give the best option's letter directly."
)


@dataclass
class EncodedChunk:
    vision_emb: torch.Tensor
    grid_thw: torch.Tensor
    chunk_index: int
    start_time: float
    end_time: float


class RecentWindowQAModel(_BaseRecentWindowQAModel):
    """Qwen2.5-VL wrapper with recent-window frame selection."""

    @torch.inference_mode()
    def encode_vision(self, frames: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode frames with the Qwen2.5-VL image-feature path."""
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

        result = self._get_multimodal_model().get_image_features(pixel_values, image_grid_thw)
        if isinstance(result, tuple):
            image_embeds = torch.cat(result[0], dim=0)
        else:
            image_embeds = result
        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return image_embeds, image_grid_thw

    @torch.inference_mode()
    def generate_with_cached_vision(
        self,
        cached_embeds: torch.Tensor,
        cached_grid_thw: torch.Tensor,
        question: str,
    ) -> str:
        """Generate from encoded visual tokens using one continuous vision prefix."""
        text_device = self._get_text_input_device()
        tokenizer = self.processor.tokenizer
        multimodal_model = self._get_multimodal_model()
        text_model = self._get_text_model()

        num_vision_tokens = int(cached_embeds.shape[0])
        self._last_num_vision_tokens = num_vision_tokens
        self._last_num_vision_frames = int(cached_grid_thw.shape[0]) if cached_grid_thw is not None else 0

        grid_rows = cached_grid_thw.to(text_device)
        expected_tokens = int((grid_rows.prod(dim=-1) // (self.merge_size**2)).sum().item())
        if expected_tokens != num_vision_tokens:
            raise ValueError(
                "cached vision token count mismatch: "
                f"embeds={num_vision_tokens} vs grid={expected_tokens}"
            )

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        input_ids_list: list[int] = []
        input_ids_list.append(self._im_start_id)
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        input_ids_list.append(self._vision_start_id)
        input_ids_list.extend([self.image_token_id] * num_vision_tokens)
        input_ids_list.append(self._vision_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend(question_ids)
        input_ids_list.append(self._im_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.append(self._im_start_id)
        input_ids_list.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=text_device)
        attention_mask = torch.ones_like(input_ids)

        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        cached_embeds = cached_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, cached_embeds)

        position_ids, _ = multimodal_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=grid_rows.to(inputs_embeds.device),
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return self._generate_from_model_inputs(
            prompt_length=int(input_ids.shape[1]),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


def build_qwen25_prompt(task: str, anno: dict[str, Any], index: int = 0) -> str:
    if task in BACKWARD_TASKS or task in REAL_TIME_TASKS:
        options = anno["options"]
        opts_str = "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options)) + ";"
        return QWEN25_BR_PROMPT_TEMPLATE.format(question=anno["question"], options=opts_str)
    return build_ovo_prompt(task, anno, index=index)


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

    encoded_window: deque[EncodedChunk] = deque(maxlen=max(1, int(recent_frames_only)))
    for chunk in chunks:
        vision_emb, grid_thw = qa.encode_vision(chunk.frames)
        encoded_window.append(
            EncodedChunk(
                vision_emb=vision_emb,
                grid_thw=grid_thw,
                chunk_index=chunk.chunk_index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
        )

    t0 = time.perf_counter()
    combined_embeds, combined_grid_thw = _combine_window_embeddings(encoded_window, qa.model.device)
    answer = qa.generate_with_cached_vision(combined_embeds, combined_grid_thw, prompt)
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
            prompt=build_qwen25_prompt(anno["task"], anno),
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
            prompt=build_qwen25_prompt(anno["task"], anno, index=index),
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
