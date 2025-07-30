# app/services/utils.py

import cv2
import numpy as np
from fastapi import UploadFile, HTTPException, status
import re
import logging
import base64
import binascii
from typing import Dict, Optional
import traceback
import hashlib
import warnings
from app.config import settings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

async def read_imagefile(file: UploadFile) -> np.ndarray:
    """
    Read and validate uploaded image file with comprehensive error handling
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Image as numpy array
        
    Raises:
        HTTPException: If file is invalid or cannot be processed
    """
    try:
        # Validate file object
        if not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Check file size (basic validation)
        if hasattr(file, 'size') and file.size is not None:
            if file.size == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty file provided"
                )
            # Max file size check (e.g., 10MB)
            max_size = 10 * 1024 * 1024  # 10MB
            if file.size > max_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
                )
        
        # Check content type if available
        if hasattr(file, 'content_type') and file.content_type:
            allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp']
            if file.content_type not in allowed_types:
                logger.warning(f"Unexpected content type: {file.content_type}")
                # Don't block processing, just log warning as content-type can be unreliable
        
        # Read file contents
        contents = await file.read()
        if not contents:
            logger.error(f"Failed to read file contents: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not read file contents"
            )
        
        if not contents or len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Convert to numpy array
        try:
            arr = np.frombuffer(contents, dtype=np.uint8)
            if arr.size == 0:
                raise ValueError("Empty array after conversion")
        except Exception as e:
            logger.error(f"Failed to convert file to numpy array: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format"
            )
        
        # Decode image using OpenCV
        try:
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("OpenCV could not decode the image")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode image. Please ensure it's a valid image file (JPEG, PNG, BMP, TIFF, WebP)"
            )
        
        # Validate decoded image
        if img.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Decoded image has zero size"
            )
        
        # Check image dimensions
        height, width = img.shape[:2]
        if height < 50 or width < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image too small: {width}x{height}. Minimum size is 50x50 pixels"
            )
        
        if height > 10000 or width > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image too large: {width}x{height}. Maximum size is 10000x10000 pixels"
            )
        
        logger.info(f"Successfully loaded image: {width}x{height}, channels: {img.shape[2] if len(img.shape) > 2 else 1}")
        return img
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading image file: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing file"
        )
    

def normalize_key(key: str) -> str:
    # Normalize key: lowercased, no spaces, underscores only
    return re.sub(r'\s+', '_', key.strip().lower())

def normalize_fields(card_type: str, raw_fields: Dict[str, str]) -> Dict[str, Optional[str]]:
    # Preprocess raw keys
    normalized_raw = {normalize_key(k): v for k, v in raw_fields.items()}
    
    mapping = settings.FIELD_MAPPINGS.get(card_type.lower(), {})
    result = {}

    for target_key, variants in mapping.items():
        for variant in variants:
            variant_key = normalize_key(variant)
            if variant_key in normalized_raw:
                result[target_key] = normalized_raw[variant_key]
                break
        else:
            result[target_key] = None  # fallback

    return result


def hash_image(image: np.ndarray) -> str:
    if image is None:
        return "None image"
    return hashlib.md5(image.tobytes()).hexdigest()

def resize_image(img: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
    """
    Resize and pad an image to fit within target dimensions while maintaining aspect ratio.
    """
    if img is None:
        logger.error("Attempted to resize a None image.")
        return None

    h, w = img.shape[:2]
    target_w, target_h = target_size

    # Calculate scaling factor to fit the image within the target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target_size
    # Assuming the input image is BGR, create a 3-channel black image for padding
    padded_img = np.full((target_h, target_w, 3), 0, dtype=np.uint8) # Black background

    # Calculate padding to center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Place the resized image into the center of the padded image
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    
    logger.info(f"Image resized from {w}x{h} to {new_w}x{new_h} and padded to {target_w}x{target_h}")
    return padded_img

def clean_text(text: str, field_name: str = "", card_type: str = "") -> str:
    """
    Clean extracted OCR text based on field type and card type with improved error handling
    
    Args:
        text: Raw OCR text
        field_name: Name of the field
        card_type: Type of card
        
    Returns:
        Cleaned text
    """
    try:
        if not text or not isinstance(text, str):
            return ""

        text = text.strip()
        if not text:
            return ""
        
        # PAN-specific cleaning
        if card_type == "pan":
            if "pan" in field_name.lower() or "number" in field_name.lower():
                return _clean_pan_number(text)
            elif "name" in field_name.lower():
                return _clean_name_field(text)
        
        # General cleaning
        return _clean_general_text(text)
        
    except Exception as e:
        logger.error(f"Error cleaning text '{text}' for field '{field_name}': {e}")
        # Return original text if cleaning fails
        return text.strip() if isinstance(text, str) else ""

def _clean_pan_number(text: str) -> str:
    """Extract PAN number using regex pattern with improved validation"""
    try:
        if not text:
            return ""
        
        # Remove common noise characters
        text = re.sub(r'[^\w\s]', '', text.upper())
        
        # PAN format: 5 letters, 4 digits, 1 letter
        pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
        match = re.search(pan_pattern, text)
        
        if match:
            pan_number = match.group()
            # Additional validation - check if it looks like a real PAN
            if len(pan_number) == 10:
                return pan_number
        
        # Fallback to general cleaning
        return _clean_general_text(text)
        
    except Exception as e:
        logger.error(f"Error cleaning PAN number '{text}': {e}")
        return _clean_general_text(text)

def _clean_name_field(text: str) -> str:
    """Clean name fields by removing common prefixes with improved handling"""
    try:
        if not text:
            return ""
        
        prefixes_to_remove = [
            "income tax department", "income tax", "permanent account number",
            "name:", "father's name:", "father name:", "fathers name:", "/",
            "FATHER'S NAME", "NAME ", "card", "department", "tax", "Name", "Father's Name"
        ]
        
        text_lower = text.lower().strip()
        cleaned_text = text
        
        for prefix in prefixes_to_remove:
            if prefix.lower() in text_lower:
                # Use case-insensitive replacement
                pattern = re.escape(prefix)
                cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                break
        
        # Remove extra whitespace and capitalize properly
        cleaned_text = ' '.join(cleaned_text.split())
        
        # Only uppercase if it looks like a name (not mixed case)
        if cleaned_text.isupper() or cleaned_text.islower():
            cleaned_text = cleaned_text.upper()
        
        return cleaned_text if cleaned_text else ""
        
    except Exception as e:
        logger.error(f"Error cleaning name field '{text}': {e}")
        return text.strip() if text else ""

def _clean_general_text(text: str) -> str:
    """General text cleaning function with improved error handling"""
    try:
        if not text:
            return ""
        
        text = text.strip()
        
        text_lower = text.lower()
        
        # Define regex pattern to remove known prefixes
        pattern = r"^(pan\s*(no)?|pan number|permanent account number(\s*card)?|name|father('?s)? name|aadh?aar(\s*no)?|dob|date of birth|birth date|address|license(\s*no)?|dl no|voter(\s*id)?|epic no)\s*[:\-.,;]?\s*"
        
        cleaned_text = re.sub(pattern, "", text_lower, flags=re.IGNORECASE).strip()

        # Remove trailing separators and non-alphanumeric noise
        cleaned_text = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned_text).strip()

        return cleaned_text if cleaned_text else text.strip()
        
    except Exception as e:
        logger.error(f"Error in general text cleaning '{text}': {e}")
        return text.strip() if text else ""

def filter_voter_fields(fields: Dict[str, str]) -> Dict[str, str]:
    """Filter and validate voter card fields with improved error handling"""
    try:
        if not fields or not isinstance(fields, dict):
            return {}
        
        # Check for back side detection
        back_fields = ['card_voterid_1_back', 'card_voterid_2_back']
        if any(field in fields for field in back_fields):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Please upload the front side of the voter ID card",
                    "card_type": "voter_back_detected"
                }
            )
        
        # Keep only allowed fields
        allowed_fields = ['age', 'father', 'gender', 'name', 'voter_id']
        filtered_fields = {}
        
        for field_name, field_value in fields.items():
            try:
                if (field_name in allowed_fields and 
                    field_value and 
                    isinstance(field_value, str) and 
                    field_value.strip()):
                    filtered_fields[field_name] = field_value.strip()
            except Exception as e:
                logger.warning(f"Error processing voter field '{field_name}': {e}")
                continue
        
        return filtered_fields
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error filtering voter fields: {e}")
        return fields  # Return original fields if filtering fails

def validate_main_id_field(fields: Dict[str, str], card_type: str) -> bool:
    """Simple validation - just check if any field has 10+ character value"""
    try:
        if not fields:
            return False
        
        for field_name, field_value in fields.items():
            if field_value and len(field_value.strip()) >= 10:
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error validating main ID field: {e}")
        return True  # Don't block on errors

def decode_base64_image(base64_data: str) -> np.ndarray:
    """
    Decode base64 image data to numpy array with comprehensive validation
    
    Args:
        base64_data: Base64 encoded image string (already validated by Pydantic)
        
    Returns:
        Decoded image as numpy array
        
    Raises:
        HTTPException: If decoding fails
    """
    logger.info(f"Request base64 hash: {hashlib.md5(base64_data.encode()).hexdigest()}")
    try:
        original_data = base64_data.strip()
        
        # Handle data URL format (double-check even though Pydantic validates)
        if original_data.startswith('data:'):
            # Extract base64 part from data URL
            if ',' in original_data:
                base64_data = original_data.split(',', 1)[1]
            else:
                raise ValueError("Invalid data URL format - missing comma separator")
        else:
            base64_data = original_data
        
        # Clean any remaining whitespace
        base64_data = re.sub(r'\s+', '', base64_data)
        
        # Decode base64 to bytes
        try:
            image_bytes = base64.b64decode(base64_data, validate=True)
        except binascii.Error as e:
            raise ValueError(f"Base64 decoding failed: {str(e)}")
        
        # Additional size validation
        if len(image_bytes) < 100:
            raise ValueError("Decoded image data too small")
        
        # Convert bytes to numpy array
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        
        if arr.size == 0:
            raise ValueError("Empty image data after conversion")
        
        # Decode image using OpenCV
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("OpenCV could not decode the image - unsupported format or corrupted data")
        
        # Validate decoded image properties
        if img.size == 0:
            raise ValueError("Decoded image has zero size")
        
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        logger.debug(f"Decoded image hash: {hash_image(img)}")

        
        # Check reasonable image dimensions
        height, width = img.shape[:2]
        if height < 50 or width < 50:
            raise ValueError(f"Image too small: {width}x{height}. Minimum size is 50x50 pixels")
        
        if height > 10000 or width > 10000:
            raise ValueError(f"Image too large: {width}x{height}. Maximum size is 10000x10000 pixels")
        
        logger.info(f"Successfully decoded image: {width}x{height}, channels: {img.shape[2] if len(img.shape) > 2 else 1}")
        return img
    
        
    except ValueError as ve:
        logger.error(f"Validation error in base64 decode: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation failed: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in base64 decode: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process image: {str(e)}"
        )

def validate_image_for_ocr(img: np.ndarray) -> bool:
    """
    Additional validation to check if image is suitable for OCR processing
    
    Args:
        img: Decoded image array
        
    Returns:
        True if image passes OCR suitability checks
    """
    try:
        # Check if image is too dark or too bright
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 10:
            logger.warning("Image appears too dark for OCR")
            return False
        
        if mean_brightness > 245:
            logger.warning("Image appears too bright/overexposed for OCR")
            return False
        
        # Check if image has sufficient contrast
        std_brightness = np.std(gray)
        if std_brightness < 5:
            logger.warning("Image has very low contrast")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in OCR image validation: {e}")
        return True  # Don't block on validation errors
    