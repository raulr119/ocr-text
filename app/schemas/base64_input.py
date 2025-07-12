from pydantic import BaseModel, Field, field_validator, ConfigDict
import base64
import binascii
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Base64ImageInput(BaseModel):
    image_data: str = Field(
        ...,
        description="Base64 encoded image data (with or without data URL prefix). Supports formats: JPEG, PNG, BMP, TIFF, WebP",
        min_length=1
    )
    
    @field_validator('image_data')
    @classmethod
    def validate_base64(cls, v):
        """
        Validate base64 image data with comprehensive error handling
        """
        try:
            if not v or not v.strip():
                logger.error("Empty image data provided")
                raise ValueError('Image data cannot be empty')
            
            original_data = v.strip()
            logger.info(f"Validating base64 data of length: {len(original_data)}")
            
            # Handle data URL format
            if original_data.startswith('data:'):
                # Validate data URL format with complete regex
                data_url_pattern = r'^data:image\/(jpeg|jpg|png|bmp|tiff|tif|webp);base64,(.+)$'
                match = re.match(data_url_pattern, original_data, re.IGNORECASE)
                
                if not match:
                    logger.error("Invalid data URL format")
                    raise ValueError(
                        'Invalid data URL format. Expected format: data:image/{format};base64,{data} '
                        'where format is one of: jpeg, jpg, png, bmp, tiff, tif, webp'
                    )
                
                # Extract the base64 data part
                base64_data = match.group(2)
                
                # Validate the image format
                image_format = match.group(1).lower()
                supported_formats = ['jpeg', 'jpg', 'png', 'bmp', 'tiff', 'tif', 'webp']
                if image_format not in supported_formats:
                    logger.error(f"Unsupported image format: {image_format}")
                    raise ValueError(f'Unsupported image format: {image_format}. Supported formats: {", ".join(supported_formats)}')
                
                logger.info(f"Valid data URL format detected: {image_format}")
            else:
                # Raw base64 data
                base64_data = original_data
                logger.info("Raw base64 data detected")
            
            # Clean any whitespace from base64 data
            base64_data = re.sub(r'\s+', '', base64_data)
            
            # Validate base64 format
            if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', base64_data):
                logger.error("Invalid base64 characters detected")
                raise ValueError('Invalid base64 characters detected')
            
            # Check base64 padding
            if len(base64_data) % 4 != 0:
                logger.error(f"Invalid base64 padding. Length: {len(base64_data)}")
                raise ValueError('Invalid base64 padding')
            
            # Try to decode to verify it's valid base64
            try:
                decoded_bytes = base64.b64decode(base64_data, validate=True)
                logger.info(f"Base64 decoded successfully. Size: {len(decoded_bytes)} bytes")
            except (binascii.Error, ValueError) as decode_error:
                logger.error(f"Base64 decoding failed: {str(decode_error)}")
                raise ValueError(f'Invalid base64 encoding: {str(decode_error)}')
            
            # Basic size validation
            if len(decoded_bytes) < 100:
                logger.error(f"Image too small: {len(decoded_bytes)} bytes")
                raise ValueError('Image data too small (minimum 100 bytes required)')
            
            # Maximum size validation (e.g., 50MB)
            max_size = 50 * 1024 * 1024  # 50MB
            if len(decoded_bytes) > max_size:
                logger.error(f"Image too large: {len(decoded_bytes)} bytes")
                raise ValueError(f'Image data too large (maximum {max_size} bytes allowed)')
            
            # Check for common image file signatures
            image_signatures = {
                b'\xff\xd8\xff': 'JPEG',
                b'\x89PNG\r\n\x1a\n': 'PNG',
                b'RIFF': 'WebP',  # WebP starts with RIFF
                b'BM': 'BMP',
                b'II*\x00': 'TIFF',
                b'MM\x00*': 'TIFF',
            }
            
            # Check if decoded data starts with a valid image signature
            valid_signature = False
            detected_format = None
            
            for signature, format_name in image_signatures.items():
                if decoded_bytes.startswith(signature):
                    valid_signature = True
                    detected_format = format_name
                    break
                # Special case for WebP - check for WebP signature after RIFF
                if signature == b'RIFF' and len(decoded_bytes) >= 12:
                    if decoded_bytes.startswith(b'RIFF') and decoded_bytes[8:12] == b'WEBP':
                        valid_signature = True
                        detected_format = 'WebP'
                        break
            
            if not valid_signature:
                logger.error("Invalid image signature detected")
                raise ValueError('Invalid image format. File does not appear to be a valid image.')
            
            logger.info(f"Valid image signature detected: {detected_format}")
            
            # Additional validation: check if image is not corrupted
            try:
                # Try to read the image header to ensure it's not corrupted
                if detected_format == 'JPEG':
                    # JPEG should end with FFD9
                    if not decoded_bytes.endswith(b'\xff\xd9'):
                        logger.warning("JPEG image may be corrupted (missing EOI marker)")
                elif detected_format == 'PNG':
                    # PNG should end with IEND chunk
                    if not decoded_bytes.endswith(b'IEND\xaeB`\x82'):
                        logger.warning("PNG image may be corrupted (missing IEND chunk)")
            except Exception as integrity_error:
                logger.warning(f"Image integrity check failed: {integrity_error}")
                # Don't raise error for integrity check failures, just log
            
            # Return the original data (with data URL prefix if it was present)
            logger.info("Base64 validation completed successfully")
            return original_data
            
        except ValueError as ve:
            # Re-raise validation errors
            logger.error(f"Validation error: {ve}")
            raise ve
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error during validation: {e}")
            raise ValueError(f'Unexpected error during image validation: {str(e)}')

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZ..."
            }
        }
    )