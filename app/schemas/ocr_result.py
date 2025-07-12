# app/schemas/ocr_result.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional

class OCRResult(BaseModel):
    status: bool = Field(..., description="Whether OCR was successful")
    message: str = Field(..., description="Status message about the OCR result")
    body: Optional[Dict[str, str]] = Field(
        default=None,
        description="Extracted text fields from the document"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": True,
                "message": "OCR extraction successful",
                "body": { "field_name": "extracted_value" } # Example body structure
            }
        }
    )