# app/routers/ocr.py

from typing import Dict
import numpy as np
import cv2
import torch
import logging

from fastapi import APIRouter, File, UploadFile, HTTPException, status

from app.services.segmentation import SegmentationService
from app.services.easyocr_service import EasyOCRService
from app.services.id_detection import IDDetectionService
from app.services.classification import ClassificationService
from app.services.utils_old import read_imagefile, clean_text, filter_voter_fields, validate_main_id_field
from app.schemas.ocr_result import OCRResult
from app.config import settings
from app.schemas.base64_input import Base64ImageInput
from app.services.utils_old import read_imagefile, clean_text, filter_voter_fields, validate_main_id_field, decode_base64_image


logger = logging.getLogger(__name__)

# Initialize services
segmentation_service = SegmentationService()
ocr_service = EasyOCRService()
id_detection_service = IDDetectionService()
classification_service = ClassificationService()

async def process_ocr_pipeline(img: np.ndarray) -> OCRResult:
    """Shared OCR processing pipeline"""
    
    # Step 1: Classification
    classified_card_type = classification_service.classify_card(img)
    if classified_card_type is None:
        logger.warning("Classification failed - no card type detected")
        return OCRResult(status=False, message="No ID card detected", body=None)

    logger.info(f"Card classified as: {classified_card_type}")

    # Step 2: Segmentation
    segmented_card = segmentation_service.crop_id_card(img)
    if segmented_card is None:
        logger.warning("Segmentation failed - could not crop card")
        return OCRResult(status=False, message="Please capture the card properly - unable to detect card boundaries", body=None)

    logger.info("Card segmented successfully")

    # Step 3: Field Detection
    detected_fields = id_detection_service.extract_fields(segmented_card, classified_card_type)
    if not detected_fields:
        logger.warning(f"No fields detected for card type: {classified_card_type}")
        return OCRResult(status=False, message="card incomplete - no readable fields detected", body=None)

    logger.info(f"Detected {len(detected_fields)} fields")

    # Step 4: OCR
    extracted_fields = {}
    for field in detected_fields:
        try:
            field_name = field["field_name"]
            field_crop = field["crop"]

            if field_crop.size == 0:
                continue

            raw_text = ocr_service.ocr_text(field_crop)
            if raw_text:
                cleaned_text = clean_text(raw_text, field_name, classified_card_type)
                if cleaned_text:
                    extracted_fields[field_name] = cleaned_text

        except Exception as e:
            logger.error(f"Error processing field {field.get('field_name', 'unknown')}: {e}")
            continue

    # Step 5: Post-processing for voter cards
    if classified_card_type == "voter":
        try:
            extracted_fields = filter_voter_fields(extracted_fields)
        except HTTPException as e:
            return OCRResult(status=False, message="Please upload the front side of the voter ID card", body=None)

    # Step 6: Validate main ID field
    if not validate_main_id_field(extracted_fields, classified_card_type):
        logger.warning(f"Main ID field missing for {classified_card_type}")
        return OCRResult(status=False, message=f"{classified_card_type} card incomplete - missing required identification field", body=None)

    if not extracted_fields:
        return OCRResult(status=False, message="No readable text found in the document", body=None)

    logger.info(f"OCR completed successfully. Extracted {len(extracted_fields)} fields")
    return OCRResult(status=True, message="OCR extraction successful", body=extracted_fields)

router = APIRouter()

@router.post("/", response_model=OCRResult, response_model_exclude_unset=True)
async def run_ocr(file: UploadFile = File(...)):
    """OCR Pipeline: Classification → Segmentation → Field Detection → OCR"""
    
    if settings.REQUIRE_GPU and not torch.cuda.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No GPU available. This service requires a GPU."
        )

    try:
        img = await read_imagefile(file)
        logger.info("Image loaded successfully")
        return await process_ocr_pipeline(img)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in OCR processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during OCR processing"
        )

@router.post("/base64", response_model=OCRResult, response_model_exclude_unset=True)
async def run_ocr_base64(request: Base64ImageInput):
    """OCR Pipeline with Base64 Input"""
    
    if settings.REQUIRE_GPU and not torch.cuda.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No GPU available. This service requires a GPU."
        )

    try:
        img = decode_base64_image(request.image_data)
        logger.info("Base64 image decoded successfully")
        return await process_ocr_pipeline(img)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in base64 OCR processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during OCR processing"
        )