from pydantic import BaseModel, Field, field_validator
import base64
import binascii
import re
from typing import Optional

class Base64ImageInput(BaseModel):
    image_data: str = Field(
        ...,
        description="Base64 encoded image data (with or without data URL prefix). Supports formats: JPEG, PNG, BMP, TIFF, WebP",
        min_length=1
    )
    
    @field_validator('image_data')
    @classmethod
    def validate_base64(cls, v):
        if not v or not v.strip():
            raise ValueError('Image data cannot be empty')
        
        original_data = v.strip()
        
        # Handle data URL format
        if original_data.startswith('data:'):
            # Validate data URL format with complete regex
            data_url_pattern = r'^data:image\/(jpeg|jpg|png|bmp|tiff|tif|webp);base64,(.+)$'
            match = re.match(data_url_pattern, original_data, re.IGNORECASE)
            
            if not match:
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
                raise ValueError(f'Unsupported image format: {image_format}. Supported formats: {", ".join(supported_formats)}')
        else:
            # Raw base64 data
            base64_data = original_data
        
        # Clean any whitespace from base64 data
        base64_data = re.sub(r'\s+', '', base64_data)
        
        # Validate base64 format
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', base64_data):
            raise ValueError('Invalid base64 characters detected')
        
        # Check base64 padding
        if len(base64_data) % 4 != 0:
            raise ValueError('Invalid base64 padding')
        
        # Try to decode to verify it's valid base64
        try:
            decoded_bytes = base64.b64decode(base64_data, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f'Invalid base64 encoding: {str(e)}')
        
        # Basic size validation
        if len(decoded_bytes) < 100:
            raise ValueError('Image data too small (minimum 100 bytes required)')
        
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
        for signature, format_name in image_signatures.items():
            if decoded_bytes.startswith(signature):
                valid_signature = True
                break
            # Special case for WebP - check for WebP signature after RIFF
            if signature == b'RIFF' and len(decoded_bytes) >= 12:
                if decoded_bytes.startswith(b'RIFF') and decoded_bytes[8:12] == b'WEBP':
                    valid_signature = True
                    break
        
        if not valid_signature:
            raise ValueError('Invalid image format. File does not appear to be a valid image.')
        
        # Return the original data (with data URL prefix if it was present)
        return original_data

    class Config:
        json_schema_extra = {
            "example": {
                "image_data": "data/image@jpeg="
            }
        }