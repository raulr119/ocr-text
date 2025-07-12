# app/routers/ocr.py

from typing import Dict
import numpy as np
import cv2
import torch
import logging
import traceback

from fastapi import APIRouter, File, UploadFile, HTTPException, status

from app.services.segmentation import SegmentationService
from app.services.easyocr_service import EasyOCRService
from app.services.id_detection import IDDetectionService
from app.services.classification import ClassificationService
from app.services.utils import (
    read_imagefile, 
    clean_text, 
    filter_voter_fields, 
    validate_main_id_field,
    decode_base64_image,
    validate_image_for_ocr,
    resize_image
)
from app.schemas.ocr_result import OCRResult
from app.schemas.base64_input import Base64ImageInput
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize services with error handling
try:
    segmentation_service = SegmentationService()
    ocr_service = EasyOCRService()
    id_detection_service = IDDetectionService()
    classification_service = ClassificationService()
    logger.info("All OCR services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OCR services: {e}")
    raise

router = APIRouter()

def check_gpu_availability():
    """Check GPU availability if required"""
    if settings.REQUIRE_GPU and not torch.cuda.is_available():
        logger.error("GPU required but not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No GPU available. This service requires a GPU."
        )

def process_ocr_pipeline(img: np.ndarray, source: str = "file") -> OCRResult:
    """
    Common OCR processing pipeline for both file and base64 inputs
    
    Args:
        img: Input image as numpy array
        source: Source type for logging ("file" or "base64")
    
    Returns:
        OCRResult object
    """
    try:
        logger.info(f"Starting OCR pipeline for {source} input")
        
        # Additional validation for OCR suitability
        if not validate_image_for_ocr(img):
            logger.warning("Image may not be suitable for OCR")
            # Don't block processing, just log warning
        

        # Step 1: Segmentation - Crop the ID card region
        try:
            segmented_card = segmentation_service.crop_id_card(img)
            if segmented_card is None:
                logger.warning("Segmentation failed - could not crop card")
                return OCRResult(
                    status=False,
                    message=f"Please capture the card properly - unable to detect card boundaries {classified_card_type}",
                    body=None
                )
            
            logger.info("Card segmented successfully")
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            return OCRResult(
                status=False,
                message="Error during card segmentation",
                body=None
            )
        # Step 2: Classification - Determine card type first
        try:
            classified_card_type = classification_service.classify_card(segmented_card)
            if classified_card_type is None:
                logger.warning("Classification failed - no card type detected")
                return OCRResult(status=False, message="No ID card detected", body=None)
            
            logger.info(f"Card classified as: {classified_card_type}")
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return OCRResult(
                status=False,
                message="Error during card classification",body=None)

        # Step 3: Field Detection - Extract field regions
        try:
            detected_fields = id_detection_service.extract_fields(segmented_card, classified_card_type)
            if not detected_fields:
                logger.warning(f"No fields detected for card type: {classified_card_type}")
                return OCRResult(status=False, message="Card incomplete - no readable fields detected", body=None)
            
            logger.info(f"Detected {len(detected_fields)} fields")
        except Exception as e:
            logger.error(f"Field detection error: {e}")
            return OCRResult(
                status=False,
                message="Error during field detection", body=None
            )

        # Step 4: OCR - Extract text from each field
        extracted_fields = {}
        failed_fields = []
        
        for field in detected_fields:
            try:
                field_name = field.get("field_name", "unknown")
                field_crop = field.get("crop")

                if field_crop is None or field_crop.size == 0:
                    logger.warning(f"Empty crop for field: {field_name}")
                    continue

                # Extract text using OCR
                raw_text = ocr_service.ocr_text(field_crop)
                if raw_text and raw_text.strip():
                    # Clean the extracted text
                    cleaned_text = clean_text(raw_text, field_name, classified_card_type)
                    if cleaned_text and cleaned_text.strip():
                        extracted_fields[field_name] = cleaned_text
                        logger.debug(f"Extracted field {field_name}: {cleaned_text}")
                    else:
                        logger.warning(f"Text cleaning failed for field: {field_name}")
                else:
                    logger.warning(f"No text extracted for field: {field_name}")

            except Exception as e:
                field_name = field.get("field_name", "unknown")
                logger.error(f"Error processing field {field_name}: {e}")
                failed_fields.append(field_name)
                continue

        if failed_fields:
            logger.warning(f"Failed to process fields: {failed_fields}")

        # Step 5: Post-processing based on card type
        if classified_card_type == "voter":
            try:
                extracted_fields = filter_voter_fields(extracted_fields)
            except HTTPException as e:
                # Handle voter back-side detection
                logger.info("Voter back-side detected")
                return OCRResult(status=False, message="Please upload the front side of the voter ID card", body=None)
            except Exception as e:
                logger.error(f"Voter field filtering error: {e}")
                return OCRResult(
                    status=False,
                    message="Error during voter field filtering",body=None
                )

        # Step 6: Validate main ID field presence
        try:
            if not validate_main_id_field(extracted_fields, classified_card_type):
                logger.warning(f"Main ID field missing for {classified_card_type}")
                return OCRResult(
                    status=False,
                    message=" card incomplete - missing required identification field",body=None
                )
        except Exception as e:
            logger.error(f"ID field validation error: {e}")
            # Don't block processing for validation errors

        logger.info(f"OCR completed successfully. Extracted {len(extracted_fields)} fields")
        
        return OCRResult(
            status=True,
            message=f"OCR extraction successful of {classified_card_type}",
            body=extracted_fields
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in OCR pipeline: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return OCRResult(
            card_type="processing_error",
            fields={"error": "Unexpected error during processing"}
        )


# def process_ocr_pipeline(img: np.ndarray, source: str = "file") -> OCRResult:
#     """
#     Common OCR processing pipeline for both file and base64 inputs

#     Args:
#         img: Input image as numpy array
#         source: Source type for logging ("file" or "base64")

#     Returns:
#         OCRResult object
#     """
#     try:
#         logger.info(f"Starting OCR pipeline for {source} input")

#         # Step 0: Resize the input image to 640x640
#         if img is not None:
#             img = resize_image(img, target_size=(640, 640))
#             if img is None:
#                 logger.error("Image resizing failed.")
#                 return OCRResult(status=False, message="Image processing failed during resizing", body=None)
#             logger.info("Image resized to 640x640")
#         else:
#             logger.error("Input image is None before resizing.")
#             return OCRResult(status=False, message="Input image not found", body=None)


#         # Additional validation for OCR suitability (after resizing)
#         if not validate_image_for_ocr(img):
#             logger.warning("Resized image may not be suitable for OCR")
#             # Don't block processing, just log warning

#         # Step 1: Classification - Determine card type first
#         try:
#             classified_card_type = classification_service.classify_card(img)
#             if classified_card_type is None:
#                 logger.warning("Classification failed - no card type detected or confidence too low")
#                 # New conditional logic: if not classified, stop here.
#                 return OCRResult(status=False, message="No ID card detected. Please capture the card properly.", body=None)

#             logger.info(f"Card classified as: {classified_card_type}")
#         except Exception as e:
#             logger.error(f"Classification error: {e}")
#             return OCRResult(
#                 status=False,
#                 message="Error during card classification",body=None)

#         # If classification is successful, proceed with the rest of the pipeline
#         # Step 2: Segmentation - Crop the ID card region
#         try:
#             segmented_card = segmentation_service.crop_id_card(img)
#             if segmented_card is None:
#                 logger.warning("Segmentation failed - could not crop card")
#                 return OCRResult(
#                     status=False,
#                     message="Please capture the card properly - unable to detect card boundaries",
#                     body=None
#                 )

#             logger.info("Card segmented successfully")
#         except Exception as e:
#             logger.error(f"Segmentation error: {e}")
#             return OCRResult(
#                 status=False,
#                 message="Error during card segmentation",
#                 body=None
#             )

#         # Step 3: Field Detection - Extract field regions
#         try:
#             detected_fields = id_detection_service.extract_fields(segmented_card, classified_card_type)
#             if not detected_fields:
#                 logger.warning(f"No fields detected for card type: {classified_card_type}")
#                 return OCRResult(status=False, message="Card incomplete - no readable fields detected", body=None)

#             logger.info(f"Detected {len(detected_fields)} fields")
#         except Exception as e:
#             logger.error(f"Field detection error: {e}")
#             return OCRResult(
#                 status=False,
#                 message="Error during field detection", body=None
#             )

#         # Step 4: OCR - Extract text from each field
#         extracted_fields = {}
#         failed_fields = []

#         for field in detected_fields:
#             try:
#                 field_name = field.get("field_name", "unknown")
#                 field_crop = field.get("crop")

#                 if field_crop is None or field_crop.size == 0:
#                     logger.warning(f"Empty crop for field: {field_name}")
#                     continue

#                 # Extract text using OCR
#                 raw_text = ocr_service.ocr_text(field_crop)
#                 if raw_text and raw_text.strip():
#                     # Clean the extracted text
#                     cleaned_text = clean_text(raw_text, field_name, classified_card_type)
#                     if cleaned_text and cleaned_text.strip():
#                         extracted_fields[field_name] = cleaned_text
#                         logger.debug(f"Extracted field {field_name}: {cleaned_text}")
#                     else:
#                         logger.warning(f"Text cleaning failed for field: {field_name}")
#                 else:
#                     logger.warning(f"No text extracted for field: {field_name}")

#             except Exception as e:
#                 field_name = field.get("field_name", "unknown")
#                 logger.error(f"Error processing field {field_name}: {e}")
#                 failed_fields.append(field_name)
#                 continue

#         if failed_fields:
#             logger.warning(f"Failed to process fields: {failed_fields}")

#         # Step 5: Post-processing based on card type
#         if classified_card_type == "voter":
#             try:
#                 extracted_fields = filter_voter_fields(extracted_fields)
#             except HTTPException as e:
#                 # Handle voter back-side detection
#                 logger.info("Voter back-side detected")
#                 return OCRResult(status=False, message="Please upload the front side of the voter ID card", body=None)
#             except Exception as e:
#                 logger.error(f"Voter field filtering error: {e}")
#                 return OCRResult(
#                     status=False,
#                     message="Error during voter field filtering",body=None
#                 )

#         # Step 6: Validate main ID field presence
#         try:
#             if not validate_main_id_field(extracted_fields, classified_card_type):
#                 logger.warning(f"Main ID field missing for {classified_card_type}")
#                 return OCRResult(
#                     status=False,
#                     message=" card incomplete - missing required identification field",body=None
#                 )
#         except Exception as e:
#             logger.error(f"ID field validation error: {e}")
#             # Don't block processing for validation errors

#         logger.info(f"OCR completed successfully. Extracted {len(extracted_fields)} fields")

#         return OCRResult(
#             status=True,
#             message="OCR extraction successful",
#             body=extracted_fields
#         )

#     except Exception as e:
#         logger.error(f"Unexpected error in OCR pipeline: {e}")
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         return OCRResult(
#             status=False,
#             message=f"Unexpected error during processing: {str(e)}",
#             body={"error": "Unexpected error during processing"}
#         )

@router.post("/", response_model=OCRResult, response_model_exclude_unset=True)
async def run_ocr(file: UploadFile = File(...)):
    """
    OCR Pipeline with File Upload: Classification ? Segmentation ? Field Detection ? OCR
    """
    try:
        # GPU availability check
        check_gpu_availability()

        # Read and validate image
        img = await read_imagefile(file)
        logger.info(f"Image loaded successfully from file: {file.filename}")

        # Process through OCR pipeline
        return process_ocr_pipeline(img, "file")
        
    except HTTPException:
        # Re-raise HTTP exceptions (from read_imagefile or GPU check)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in file OCR endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during file processing"
        )

@router.post("/base64", response_model=OCRResult, response_model_exclude_unset=True)
async def run_ocr_base64(request: Base64ImageInput):
    """
    OCR Pipeline with Base64 Input: Classification ? Segmentation ? Field Detection ? OCR
    """
    try:
        # GPU availability check
        check_gpu_availability()

        # Decode base64 image (this already includes comprehensive validation)
        img = decode_base64_image(request.image_data)
        logger.info("Base64 image decoded successfully")

        # Process through OCR pipeline
        return process_ocr_pipeline(img, "base64")
        
    except HTTPException:
        # Re-raise HTTP exceptions (from decode_base64_image or GPU check)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in base64 OCR endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during base64 processing"
        )

# Health check endpoint for the OCR router
# @router.get("/health")
# async def health_check():
#     """Health check endpoint for OCR service"""
#     try:
#         # Check if services are properly initialized
#         services_status = {
#             "segmentation_service": segmentation_service is not None,
#             "ocr_service": ocr_service is not None,
#             "id_detection_service": id_detection_service is not None,
#             "classification_service": classification_service is not None,
#             "gpu_available": torch.cuda.is_available() if settings.REQUIRE_GPU else "not_required"
#         }
        
#         return {
#             "status": "healthy",
#             "services": services_status,
#             "gpu_required": settings.REQUIRE_GPU
#         }
#     except Exception as e:
#         logger.error(f"Health check failed: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Service unhealthy"
#         )
    