# app/services/paddleocr_service.py

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from paddleocr import PaddleOCR
import paddle
import os
from dotenv import load_dotenv
import re

load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("ppocr").setLevel(logging.WARNING)

class PaddleOCRService:
    def __init__(self):
        langs_cfg = os.getenv("OCR_LANGUAGES", "en")

        if isinstance(langs_cfg, str):
            self.langs = [lang.strip() for lang in langs_cfg.split(',') if lang.strip()]
        elif isinstance(langs_cfg, list):
            self.langs = langs_cfg
        else:
            logger.warning(f"Invalid OCR_LANGUAGES format: {langs_cfg}, using default ['en']")
            self.langs = ['en']

        supported_languages = ["en", "ch", "french", "german", "hi", "mr", "ta", "bn", "gu", "kn", "ml", "or", "pa", "te", "ur"]
        valid_langs = [lang for lang in self.langs if lang in supported_languages]

        if not valid_langs:
            logger.warning("No valid languages found, using default 'en'")
            valid_langs = ['en']

        self.langs = valid_langs
        self.use_gpu = os.getenv("OCR_USE_GPU", "true").lower() in ("1", "true", "yes")
        primary_lang = self.langs[0]

        try:
            if self.use_gpu and paddle.device.is_compiled_with_cuda():
                paddle.set_device('gpu')
            else:
                paddle.set_device('cpu')

            self.reader = PaddleOCR(
                lang=primary_lang,
                use_angle_cls=True,
                show_log=False
            )
            logger.info(f"Initialized PaddleOCR with language: {primary_lang}, GPU: {self.use_gpu}")

        except Exception as e:
            logger.error(f"Error initializing PaddleOCR with language {primary_lang}: {e}")
            try:
                logger.info("Attempting fallback initialization with English only")
                self.langs = ['en']
                paddle.set_device('cpu')
                self.reader = PaddleOCR(
                    lang='en',
                    use_angle_cls=True,
                    show_log=False
                )
                logger.info("PaddleOCR initialized with English fallback")
            except Exception as fallback_error:
                logger.error(f"Fallback initialization also failed: {fallback_error}")
                raise RuntimeError(f"Failed to initialize PaddleOCR: {e}")

    def is_id_number_field(self, field_name: str, card_type: str) -> bool:
        """Check if a field is likely an ID number that needs precise extraction"""
        if not field_name or not card_type:
            return False
        
        field_lower = field_name.lower()
        
        # Define patterns for ID number fields (these are usually larger text)
        id_patterns = ['pan', 'number', 'card_number', 'aadhar', 'aadhaar', 'uid', 
                    'dl', 'license', 'licence', 'voter', 'epic']
        
        return any(pattern in field_lower for pattern in id_patterns)

    def is_small_text_field(self, field_name: str) -> bool:
        """Check if a field contains smaller text that needs gentle preprocessing"""
        if not field_name:
            return False
        
        field_lower = field_name.lower()
        small_text_patterns = ['name', 'gender', 'dob', 'date', 'birth', 'address', 
                            'father', 'mother', 'issue', 'validity', 'category']
        
        return any(pattern in field_lower for pattern in small_text_patterns)

    def gentle_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Apply very gentle preprocessing for small text fields"""
        processed = img.copy()
        
        # Only apply very mild contrast enhancement
        if len(processed.shape) == 3:
            # Convert to LAB and apply very mild CLAHE
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16,16))  # Very gentle
            l = clahe.apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16,16))
            processed = clahe.apply(processed)
        
        return processed

    def assess_image_quality(self, img: np.ndarray) -> Dict[str, float]:
        """Assess image quality metrics to determine preprocessing needs"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Calculate quality metrics
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()  # Sharpness
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)  # Contrast
        
        return {
            'sharpness': variance,
            'brightness': mean_brightness,
            'contrast': std_brightness
        }

    def minimal_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Apply minimal preprocessing that preserves character integrity"""
        processed = img.copy()
        
        # Only apply very light contrast enhancement
        if len(processed.shape) == 3:
            # Convert to LAB and apply mild CLAHE
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            l = clahe.apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        return processed

    def adaptive_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive preprocessing based on image quality"""
        processed = img.copy()
        quality = self.assess_image_quality(img)
        
        # Apply preprocessing based on quality assessment
        if quality['sharpness'] < 100:  # Blurry image
            processed = self.sharpen_image(processed)
        
        if quality['brightness'] < 100 or quality['brightness'] > 180:  # Poor lighting
            processed = self.enhance_contrast(processed)
        
        if quality['contrast'] < 30:  # Low contrast
            processed = self.adaptive_threshold(processed)
        
        return processed

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        
        return enhanced

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply mild sharpening"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.5  # Reduced intensity
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def ocr_with_fallback(self, img: np.ndarray, field_name: str = "", card_type: str = "") -> Tuple[str, float]:
        """
        OCR with smart fallback strategy based on field type
        """
        results = []
        
        # Strategy 1: ID numbers (larger text) - use original image
        if self.is_id_number_field(field_name, card_type):
            logger.debug(f"Using original image for ID field: {field_name}")
            
            # Try with original image first (works best for large text)
            try:
                result = self.reader.ocr(img, cls=True)
                text, confidence = self._extract_text_and_confidence(result)
                if text and confidence > 0.5:
                    results.append(('original', text, confidence))
            except Exception as e:
                logger.warning(f"Original image OCR failed: {e}")
        
        # Strategy 2: Small text fields (name, gender, DOB) - use gentle preprocessing
        elif self.is_small_text_field(field_name):
            logger.debug(f"Using gentle preprocessing for small text field: {field_name}")
            
            # Try with gentle preprocessing first
            try:
                processed_img = self.gentle_preprocessing(img)
                result = self.reader.ocr(processed_img, cls=True)
                text, confidence = self._extract_text_and_confidence(result)
                if text and confidence > 0.4:  # Lower threshold for small text
                    results.append(('gentle', text, confidence))
            except Exception as e:
                logger.warning(f"Gentle preprocessing failed: {e}")
            
            # Fallback to original
            try:
                result = self.reader.ocr(img, cls=True)
                text, confidence = self._extract_text_and_confidence(result)
                if text:
                    results.append(('original', text, confidence))
            except Exception as e:
                logger.warning(f"Original image OCR failed: {e}")
        
        # Strategy 3: Other fields - use current adaptive approach
        else:
            logger.debug(f"Using adaptive preprocessing for field: {field_name}")
            
            try:
                processed_img = self.adaptive_preprocessing(img)
                result = self.reader.ocr(processed_img, cls=True)
                text, confidence = self._extract_text_and_confidence(result)
                if text and confidence > 0.5:
                    results.append(('adaptive', text, confidence))
            except Exception as e:
                logger.warning(f"Adaptive preprocessing failed: {e}")
            
            # Fallback to original
            try:
                result = self.reader.ocr(img, cls=True)
                text, confidence = self._extract_text_and_confidence(result)
                if text:
                    results.append(('original', text, confidence))
            except Exception as e:
                logger.warning(f"Original image OCR failed: {e}")
        
        # Select best result
        if results:
            best_result = max(results, key=lambda x: x[2])
            logger.debug(f"Best result for {field_name}: method={best_result[0]}, confidence={best_result[2]:.2f}")
            return best_result[1], best_result[2]
        
        return "", 0.0
    def _extract_text_and_confidence(self, result) -> Tuple[str, float]:
        """Extract text and average confidence from OCR result"""
        if not result or not result[0]:
            return "", 0.0
        
        text_list = []
        confidence_sum = 0
        count = 0
        
        for line in result[0]:
            if line and len(line) > 1 and line[1] and len(line[1]) > 0:
                text_list.append(line[1][0])
                confidence_sum += line[1][1]
                count += 1
        
        if count == 0:
            return "", 0.0
        
        extracted_text = " ".join(text_list).strip()
        avg_confidence = confidence_sum / count
        
        return extracted_text, avg_confidence

    def ocr_text(self, img: np.ndarray, field_name: str = "", card_type: str = "") -> str:
        """
        Main OCR method with smart preprocessing selection
        
        Args:
            img: Input image
            field_name: Name of the field being processed
            card_type: Type of card
            
        Returns:
            Extracted text
        """
        try:
            if img is None or img.size == 0:
                logger.warning("Empty or None image provided to OCR")
                return ""
            
            logger.debug(f"OCR input image shape: {img.shape}")
            
            # Use smart fallback strategy
            text, confidence = self.ocr_with_fallback(img, field_name, card_type)
            
            # Apply post-processing
            final_text = self.post_process_text(text, field_name, card_type)
            
            logger.debug(f"OCR extracted text: '{final_text}' (confidence: {confidence:.2f})")
            return final_text
            
        except Exception as e:
            logger.error(f"Error during OCR text extraction: {e}")
            return ""

    def _post_process_small_text(self, text: str, field_name: str) -> str:
        """Post-process small text fields with gentle corrections"""
        if not text:
            return text
        
        field_lower = field_name.lower()
        
        # Gender-specific corrections
        if 'gender' in field_lower:
            text_upper = text.upper().strip()
            if 'LSML' in text_upper or 'LSMA' in text_upper:
                return 'MALE'
            elif 'FSML' in text_upper or 'FSMA' in text_upper:
                return 'FEMALE'
            # Add more gender-specific corrections as needed
        
        # Name-specific corrections
        if 'name' in field_lower:
            # Remove common OCR artifacts from names
            text = re.sub(r'[|]', 'I', text)
            text = re.sub(r'[0]', 'O', text)  # Only for names
            text = re.sub(r'\s+', ' ', text.strip())
        
        # DOB-specific corrections
        if 'dob' in field_lower or 'date' in field_lower:
            # Basic date format cleaning
            text = re.sub(r'[|]', '1', text)
            text = re.sub(r'[O]', '0', text)
        
        return text

    def post_process_text(self, text: str, field_name: str = "", card_type: str = "") -> str:
        """Post-process OCR text based on field type"""
        if not text:
            return text
        
        # ID number specific post-processing
        if self.is_id_number_field(field_name, card_type):
            return self._post_process_id_number(text, card_type)
        
        # Small text specific post-processing
        elif self.is_small_text_field(field_name):
            return self._post_process_small_text(text, field_name)
        
        # General post-processing
        return self._post_process_general(text)

    def _post_process_id_number(self, text: str, card_type: str) -> str:
        """Post-process ID numbers with format-specific corrections"""
        if not text:
            return text
        
        # Remove spaces and common noise
        clean_text = re.sub(r'\s+', '', text.upper())
        
        # Card-specific corrections
        if card_type.lower() == 'pan':
            # PAN format: 5 letters, 4 digits, 1 letter
            # Fix common OCR errors
            clean_text = re.sub(r'[0O]', 'O', clean_text[:5]) + re.sub(r'[O]', '0', clean_text[5:9]) + re.sub(r'[0-9]', '', clean_text[9:])[:1]
            
        elif card_type.lower() == 'aadhar':
            # Aadhaar: 12 digits only
            clean_text = re.sub(r'[^0-9]', '', clean_text)[:12]
            
        return clean_text

    def _post_process_general(self, text: str) -> str:
        """General post-processing for non-ID fields"""
        if not text:
            return text
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common OCR artifacts
        text = re.sub(r'[|]', 'I', text)  # Common misread
        
        return text

    def ocr_with_confidence(self, img: np.ndarray, field_name: str = "", card_type: str = "") -> Tuple[str, float]:
        """
        Extract text with confidence score using smart preprocessing
        
        Returns:
            Tuple of (extracted_text, confidence)
        """
        try:
            if img is None or img.size == 0:
                return "", 0.0
            
            text, confidence = self.ocr_with_fallback(img, field_name, card_type)
            final_text = self.post_process_text(text, field_name, card_type)
            
            return final_text, confidence
            
        except Exception as e:
            logger.error(f"Error during OCR with confidence: {e}")
            return "", 0.0