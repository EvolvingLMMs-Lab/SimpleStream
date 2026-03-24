from __future__ import annotations

import torch
from PIL import Image

from lib.recent_window_eval import (
    RecentWindowQAModel as _BaseRecentWindowQAModel,
    evaluate_ovo_backward_realtime,
    evaluate_ovo_forward,
    flatten_gathered_results,
    print_ovo_results,
)


class RecentWindowQAModel(_BaseRecentWindowQAModel):
    """Qwen3 release wrapper aligned with the cached single-block builder."""

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation,
        )
        self.vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.merge_size = self._get_visual_module().spatial_merge_size

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

        pixel_values = inputs["pixel_values"].to(self.model.device, dtype=self._get_visual_dtype())
        image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
        image_embeds = self._flatten_vision_features(
            self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
        )

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
        device = self.model.device
        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model()

        num_vision_tokens = int(cached_embeds.shape[0])
        self._last_num_vision_tokens = num_vision_tokens
        self._last_num_vision_frames = int(cached_grid_thw.shape[0]) if cached_grid_thw is not None else 0

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        grid_rows = cached_grid_thw.to(device)

        input_ids_list: list[int] = []
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        input_ids_list.append(self.vision_start_id)
        input_ids_list.extend([self.image_token_id] * num_vision_tokens)
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
        cached_embeds = cached_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, cached_embeds)

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
        cached_embeds, cached_grid_thw = self.encode_vision(frames)
        return self.generate_with_cached_vision(cached_embeds, cached_grid_thw, question)
