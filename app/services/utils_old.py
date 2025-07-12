# app/services/utils.py

import cv2
import numpy as np
from fastapi import UploadFile, HTTPException, status
import re
import logging
import base64
import binascii
import re
from typing import Dict

logger = logging.getLogger(__name__)

async def read_imagefile(file: UploadFile) -> np.ndarray:
    """Read and validate uploaded image file"""
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img
    except Exception as e:
        logger.error(f"Failed to read image file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {e}"
        )

def clean_text(text: str, field_name: str = "", card_type: str = "") -> str:
    """
    Clean extracted OCR text based on field type and card type
    
    Args:
        text: Raw OCR text
        field_name: Name of the field
        card_type: Type of card
        
    Returns:
        Cleaned text
    """
    if not text:
        return text

    text = text.strip()
    
    # PAN-specific cleaning
    if card_type == "pan":
        if "pan" in field_name.lower() or "number" in field_name.lower():
            return _clean_pan_number(text)
        elif "name" in field_name.lower():
            return _clean_name_field(text)
    
    # General cleaning
    return _clean_general_text(text)

def _clean_pan_number(text: str) -> str:
    """Extract PAN number using regex pattern"""
    pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
    match = re.search(pan_pattern, text.upper())
    return match.group() if match else _clean_general_text(text)

def _clean_name_field(text: str) -> str:
    """Clean name fields by removing common prefixes"""
    prefixes_to_remove = [
        "income tax department", "income tax", "permanent account number",
        "name:", "father's name:", "father name:", "fathers name:","/","FATHER'S NAME","NAME ",
        "card", "department", "tax"
    ]
    
    text_lower = text.lower()
    for prefix in prefixes_to_remove:
        if prefix in text_lower:
            text = re.sub(re.escape(prefix), '', text_lower).strip()
            break
    
    # Capitalize properly
    return ' '.join(text.split()).upper() if text else ""

def _clean_general_text(text: str) -> str:
    """General text cleaning function"""
    prefixes_to_remove = [
        "pan:", "pan no:", "pan number:", "permanent account number:",
        "permanent account number card", "name:", "father:", "father's name:",
        "father name:", "aadhar:", "aadhaar:", "aadhar no:", "aadhaar no:",
        "dob:", "date of birth:", "birth date:", "address:", "license:",
        "license no:", "dl no:", "voter:", "voter id:", "epic no:"
    ]
    
    text_lower = text.lower().strip()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix):
            cleaned = text[len(prefix):].strip()
            cleaned = cleaned.lstrip(":-.,;").strip()
            return cleaned
    
    return text

def filter_voter_fields(fields: Dict[str, str]) -> Dict[str, str]:
    """Filter and validate voter card fields"""
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
        if field_name in allowed_fields and field_value and field_value.strip():
            filtered_fields[field_name] = field_value.strip()
    
    return filtered_fields

def validate_main_id_field(fields: Dict[str, str], card_type: str) -> bool:
    """Check if main ID field is present and valid"""
    id_field_patterns = {
        'pan': ['pan', 'number', 'pan_number', 'pan_no'],
        'aadhar': ['aadhar', 'aadhaar', 'number', 'aadhar_number', 'aadhaar_number'],
        'driving': ['license', 'dl', 'number', 'license_number', 'dl_number'],
        'voter': ['voter_id', 'epic', 'number', 'voter_number']
    }
    
    if card_type not in id_field_patterns:
        return True  # Unknown card type, don't block
    
    patterns = id_field_patterns[card_type]
    
    # Check if any field contains the ID patterns
    for field_name, field_value in fields.items():
        field_name_lower = field_name.lower()
        for pattern in patterns:
            if pattern in field_name_lower and field_value and field_value.strip():
                return True
    
    return False


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