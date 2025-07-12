# app/services/easyocr_service.py

import cv2
import easyocr
import numpy as np
from typing import List
import logging
from app.config import settings
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class EasyOCRService:
    def __init__(self):
        """Initialize EasyOCR service using config settings"""
        # Fix: Properly handle language configuration
        if isinstance(settings.OCR_LANGUAGES, str):
            # If it's a string, split by comma and strip whitespace
            self.langs = [lang.strip() for lang in settings.OCR_LANGUAGES.split(',') if lang.strip()]
        elif isinstance(settings.OCR_LANGUAGES, list):
            # If it's already a list, use it directly
            self.langs = settings.OCR_LANGUAGES
        else:
            # Fallback to default languages
            logger.warning(f"Invalid OCR_LANGUAGES format: {settings.OCR_LANGUAGES}, using default ['en']")
            self.langs = ['en']
        
        # Validate language codes
        supported_languages = [
            'en', 'hi', 'zh', 'ja', 'ko', 'th', 'vi', 'ar', 'ru', 'es', 'fr', 'de', 'pt', 'it',
            'nl', 'pl', 'sv', 'da', 'no', 'fi', 'cs', 'sk', 'hu', 'ro', 'bg', 'hr', 'sr', 'uk',
            'be', 'lt', 'lv', 'et', 'mt', 'ga', 'cy', 'is', 'mk', 'sq', 'az', 'uz', 'mn', 'ne',
            'si', 'km', 'lo', 'my', 'ka', 'am', 'te', 'kn', 'ml', 'ta', 'bn', 'as', 'or', 'gu',
            'pa', 'ur', 'fa', 'ps', 'dv', 'si', 'bo', 'dz', 'ky', 'tg', 'tk', 'kk', 'hy'
        ]
        
        # Filter out unsupported languages
        valid_langs = [lang for lang in self.langs if lang in supported_languages]
        if not valid_langs:
            logger.warning("No valid languages found, using default 'en'")
            valid_langs = ['en']
        elif len(valid_langs) != len(self.langs):
            invalid_langs = [lang for lang in self.langs if lang not in supported_languages]
            logger.warning(f"Unsupported languages removed: {invalid_langs}")
        
        self.langs = valid_langs
        self.use_gpu = settings.OCR_USE_GPU
        self.preprocess = settings.OCR_PREPROCESS
        self.upscale_factor = settings.OCR_UPSCALE_FACTOR
        
        try:
            self.reader = easyocr.Reader(self.langs, gpu=self.use_gpu)
            logger.info(f"Initialized EasyOCR with languages: {self.langs}, GPU: {self.use_gpu}")
        except Exception as e:
            logger.error(f"Error initializing EasyOCR with languages {self.langs}: {e}")
            # Try with just English as fallback
            try:
                logger.info("Attempting fallback initialization with English only")
                self.langs = ['en']
                self.reader = easyocr.Reader(self.langs, gpu=self.use_gpu)
                logger.info("EasyOCR initialized with English fallback")
            except Exception as fallback_error:
                logger.error(f"Fallback initialization also failed: {fallback_error}")
                raise RuntimeError(f"Failed to initialize EasyOCR: {e}")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image by upscaling for better OCR accuracy"""
        h, w = img.shape[:2]
        img_up = cv2.resize(img, (w * self.upscale_factor, h * self.upscale_factor), 
                           interpolation=cv2.INTER_CUBIC)
        return img_up

    def ocr_text(self, img: np.ndarray) -> str:
        """
        Extract text from image
        
        Args:
            img: Input image array
            
        Returns:
            Concatenated text string
        """
        try:
            # Apply preprocessing if enabled
            img_input = self._preprocess(img) if self.preprocess else img
            
            # Run EasyOCR
            # result = self.reader.readtext(img, detail=0)
            result = self.reader.readtext(img_input, detail=0)
            text = " ".join(result) if result else ""
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return ""