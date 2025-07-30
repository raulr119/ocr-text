# app/services/segmentation.py

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional
import logging
from app.config import settings
import warnings
warnings.filterwarnings("ignore")
import os
import sys

logger = logging.getLogger(__name__)

class SegmentationService:
    def __init__(self):
        """Initialize segmentation service using config settings"""
        self.model_path = settings.SEGMENTATION_MODEL_PATH
        # logger.info(f"Segmentation model path from settings: {self.model_path}")

        self.use_mask = settings.SEGMENTATION_USE_MASK
        self.conf_threshold = settings.SEGMENTATION_CONFIDENCE_THRESHOLD
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Segmentation model file not found at: {self.model_path}")

            self.model = YOLO(self.model_path)
            # logger.info(f"L oaded segmentation model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading segmentation model: {e}")
            raise RuntimeError(f"Failed to load segmentation model: {e}")

    def crop_id_card(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Segment and crop ID card from image
        
        Args:
            image: Input color image array (BGR)
            
        Returns:
            Cropped image of the ID card, or None if no detection meets threshold
        """
        try:
            # Run prediction
            results = self.model.predict(source=image.copy(), conf=0.3, show=False)
            res = results[0]

            # Extract boxes and confidences
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                logger.info("No segmentation detections found")
                return None

            confs = boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            best_conf = float(confs[best_idx])

            if best_conf < self.conf_threshold:
                logger.info(f"Detection confidence {best_conf:.2f} below threshold {self.conf_threshold}")
                return None

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes.xyxy.cpu().numpy()[best_idx])

            if not self.use_mask or res.masks is None:
                # Use bounding box crop
                logger.info(f"Cropping by box at confidence {best_conf:.2f}")
                return image[y1:y2, x1:x2]

            # Use mask-based crop
            logger.info(f"Cropping by mask at confidence {best_conf:.2f}")
            mask = res.masks.data[best_idx].cpu().numpy().astype(np.uint8)
            
            # Resize mask to image size if needed
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Apply mask and crop
            masked = cv2.bitwise_and(image, image, mask=mask)
            return masked[y1:y2, x1:x2]
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            return None