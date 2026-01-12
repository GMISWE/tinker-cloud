"""
VLM Image Processor

CPU-only image processing for Vision-Language Models.
Converts base64-encoded images to pixel_values tensors using HuggingFace processors.
"""
import base64
import io
import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class VLMImageProcessor:
    """Process images for VLM training. CPU-only, no GPU required.

    This class handles the conversion of base64-encoded images to processed
    tensors (pixel_values, image_grid_thw, etc.) that Miles/Megatron expects
    for VLM training.

    The HuggingFace processor performs:
    - Image decoding (CPU)
    - Resize/normalize (CPU)
    - Create pixel_values tensor (CPU numpy -> CPU torch tensor)

    GPU is only needed for the vision encoder forward pass inside Miles.
    """

    _processor_cache: Dict[str, Any] = {}

    @classmethod
    def get_processor(cls, model_name: str) -> Any:
        """Lazy-load and cache the HuggingFace processor.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-VL-2B-Instruct")

        Returns:
            HuggingFace processor instance
        """
        if model_name not in cls._processor_cache:
            try:
                from transformers import AutoProcessor
                logger.info(f"Loading VLM processor for {model_name}")
                cls._processor_cache[model_name] = AutoProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                )
                logger.info(f"VLM processor loaded successfully for {model_name}")
            except Exception as e:
                logger.error(f"Failed to load processor for {model_name}: {e}")
                raise
        return cls._processor_cache[model_name]

    @classmethod
    def decode_image(cls, image_data: str) -> Image.Image:
        """Decode a base64-encoded image to PIL Image.

        Args:
            image_data: Base64-encoded image string

        Returns:
            PIL Image in RGB mode
        """
        try:
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return pil_image
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise ValueError(f"Invalid image data: {e}")

    @classmethod
    def process_images(
        cls,
        image_data_list: List[str],
        model_name: str
    ) -> Dict[str, torch.Tensor]:
        """Convert base64 images to pixel_values tensors (CPU).

        Args:
            image_data_list: List of base64-encoded image strings
            model_name: HuggingFace model name for processor selection

        Returns:
            Dict with processed tensors, typically including:
            - pixel_values: Processed image tensor
            - image_grid_thw: Grid dimensions for Qwen VL models
            (excludes input_ids and attention_mask which come from text)
        """
        if not image_data_list:
            return {}

        processor = cls.get_processor(model_name)

        # Decode base64 to PIL images
        pil_images = []
        for i, b64_data in enumerate(image_data_list):
            try:
                pil_image = cls.decode_image(b64_data)
                pil_images.append(pil_image)
                logger.debug(f"Decoded image {i}: {pil_image.size}")
            except Exception as e:
                logger.error(f"Failed to decode image {i}: {e}")
                raise

        # Process to tensors (CPU operation)
        try:
            # Use processor with images only - we don't want text processing here
            # Different processors have different APIs, try common patterns
            if hasattr(processor, 'image_processor'):
                # Some processors have a separate image_processor
                processed = processor.image_processor(
                    images=pil_images,
                    return_tensors="pt"
                )
            else:
                # Standard processor call - exclude text to avoid tokenization
                processed = processor(
                    images=pil_images,
                    return_tensors="pt"
                )
        except Exception as e:
            logger.error(f"Processor failed: {e}")
            raise

        # Return only image-related tensors (exclude text-related ones)
        result = {}
        for k, v in processed.items():
            if k not in ["input_ids", "attention_mask", "token_type_ids"]:
                if isinstance(v, torch.Tensor):
                    result[k] = v
                    logger.debug(f"Processed tensor {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    # Some values might be lists or other types
                    result[k] = v
                    logger.debug(f"Processed value {k}: type={type(v)}")

        logger.info(f"Processed {len(pil_images)} images -> {list(result.keys())}")
        return result

    @classmethod
    def process_batch(
        cls,
        batch_image_data: List[List[str]],
        model_name: str
    ) -> List[Dict[str, torch.Tensor]]:
        """Process images for a batch of samples.

        Args:
            batch_image_data: List of image data lists (one per sample)
            model_name: HuggingFace model name

        Returns:
            List of processed tensor dicts (one per sample)
        """
        results = []
        for i, image_data_list in enumerate(batch_image_data):
            if image_data_list:
                try:
                    processed = cls.process_images(image_data_list, model_name)
                    results.append(processed)
                except Exception as e:
                    logger.error(f"Failed to process images for sample {i}: {e}")
                    results.append({})
            else:
                results.append({})
        return results

    @classmethod
    def clear_cache(cls):
        """Clear the processor cache to free memory."""
        cls._processor_cache.clear()
        logger.info("VLM processor cache cleared")
