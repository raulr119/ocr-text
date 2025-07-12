# app/services/id_detection.py

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional
import logging
from app.config import settings
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class IDDetectionService:
    def __init__(self):
        """Initialize ID detection service using config settings"""
        self.model_paths = settings.FIELD_DETECTION_MODELS
        self.conf_threshold = settings.FIELD_DETECTION_CONFIDENCE_THRESHOLD
        
        # Load YOLO models for each card type
        self.models: Dict[str, YOLO] = {}
        self.class_names: Dict[str, List[str]] = {}
        
        for card_type, path in self.model_paths.items():
            try:
                model = YOLO(path)
                self.models[card_type] = model
                # logger.info(f"Loaded field detection model for '{card_type}' from {path}")

                # Override class names if needed
                if card_type.lower() in settings.FIELD_LABELS:
                    names_list = settings.FIELD_LABELS[card_type.lower()]

                else:
                    names = model.names
                    if isinstance(names, dict):
                        max_idx = max(names.keys())
                        names_list = [names.get(i, f'field_{i}') for i in range(max_idx + 1)]
                    else:
                        names_list = list(names)

                self.class_names[card_type] = names_list


            except Exception as e:
                logger.error(f"Error loading model for '{card_type}': {e}")

    def _get_field_name(self, card_type: str, cls: int) -> str:
        """Get human-readable field name from class index"""
        names_list = self.class_names.get(card_type, [])
        if 0 <= cls < len(names_list):
            return names_list[cls] # This retrieves the name from the `names_list`
        return f"field_{cls}"
    def extract_fields(self, image: np.ndarray, card_type: str) -> List[Dict]:
        """
        Extract fields from segmented card image
        
        Args:
            image: Cropped card image
            card_type: Type of card for field extraction
            
        Returns:
            List of detected fields with metadata
        """
        if card_type not in self.models:
            logger.error(f"No model loaded for card type '{card_type}'")
            return []

        try:
            model = self.models[card_type]
            results = model.predict(source=image.copy(), conf=self.conf_threshold, show=False)
            # logger.debug(f"Image hash: {hash_image(image)}")
            boxes = results[0].boxes
            fields = []
            
            if boxes is not None:
                for box, conf, cls in zip(boxes.xyxy.cpu().numpy(),
                                          boxes.conf.cpu().numpy(),
                                          boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    crop = image[y1:y2, x1:x2]
                    field_name = self._get_field_name(card_type, int(cls))
                    
                    fields.append({
                        'field_name': field_name,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'crop': crop
                    })
            
            logger.info(f"Extracted {len(fields)} fields for '{card_type}'")
            return fields
            
        except Exception as e:
            logger.error(f"Error extracting fields for '{card_type}': {e}")
            return []

    def _get_field_name(self, card_type: str, cls: int) -> str:
        """Get human-readable field name from class index"""
        names_list = self.class_names.get(card_type, [])
        if 0 <= cls < len(names_list):
            return names_list[cls]
        return f"field_{cls}"