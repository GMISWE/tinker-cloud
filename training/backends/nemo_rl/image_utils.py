import logging
from typing import List, Dict
import torch
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)
class ImagePreprocessor:
    def __init__(self, hf_path: str):
        self.hf_path = hf_path
        self._processor = None
        self._multimodal_keys = None

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.hf_path,trust_remote_code=True)
            logger.info(f"Loaded image processor from {self.hf_path}")
        return self._processor

    @property
    def multimodal_keys(self) -> list[str]:
        if self._multimodal_keys is None:
            from nemo_rl.data.multimodal_utils import get_multimodal_keys_from_processor
            self._multimodal_keys = get_multimodal_keys_from_processor(self.processor)
            logger.info("Multimodal keys for %s: %s", self.hf_path, self._multimodal_keys)
        return self._multimodal_keys
    
    def process_images(self, image_bytes_list: List[bytes]) -> Dict[str, torch.Tensor]:
        if not image_bytes_list:
            return {}
        
        pil_images = []
        for img_bytes in image_bytes_list:
            pil_images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))
        
        processed = self.processor.image_processor(images=pil_images, return_tensors="pt")
        result = {}
        for key in self.multimodal_keys:
            if key in processed:
                result[key] = processed[key]
        return result
