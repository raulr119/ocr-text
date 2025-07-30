from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class OCRResultBody(BaseModel):
    card_type_id: int
    card_type: str
    name: Optional[str] = None
    id_number: Optional[str] = None
    dob: Optional[str] = None
    address: Optional[str] = None
    gender: Optional[str] = None
    expiry_date: Optional[str] = None
    coname: Optional[str] = None

class OCRResult(BaseModel):
    status: bool = Field(..., description="Whether OCR was successful")
    message: str = Field(..., description="Status message about the OCR result")
    body: Optional[OCRResultBody] = Field(
        default=None,
        description="Extracted structured fields from the document"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": True,
                "message": "OCR extraction successful",
                "body": {
                    "card_type": "aadhar",
                    "name": "John Doe",
                    "id_number": "1234 5678 9012",
                    "dob": "1990-01-01",
                    "address": "123 Main St",
                    "gender": "Male"
                }
            }
        }
    )
