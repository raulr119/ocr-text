from typing import Dict
import numpy as np
import cv2
import torch
import logging
import traceback
import uuid
import gc
import psutil
import os
import hashlib
import warnings
warnings.filterwarnings("ignore")

from fastapi import APIRouter, File, UploadFile, HTTPException, status

from app.services.segmentation import SegmentationService

# from app.services.easyocr_service import EasyOCRService
from app.services.paddleocr_service import PaddleOCRService
from app.services.id_detection import IDDetectionService
from app.services.classification import ClassificationService
from app.services.utils import (
    read_imagefile, 
    clean_text, 
    filter_voter_fields, 
    validate_main_id_field,
    decode_base64_image,
    validate_image_for_ocr,
    resize_image,
    hash_image,
    normalize_fields
)
from app.schemas.ocr_result import OCRResult, OCRResultBody
from app.schemas.base64_input import Base64ImageInput
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# def check_gpu_availability():
#     """Check GPU availability if required"""
#     if settings.OCR_USE_GPU and not torch.cuda.is_available():
#         logger.error("GPU required but not available")
#         torch.cuda.empty_cache()
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="No GPU available. This service requires a GPU."
#         )

def cleanup_memory():
    """Clean up memory for both GPU and CPU systems"""
    # GPU cleanup (safe on CPU systems)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # CPU cleanup
    gc.collect()  # Force garbage collection
    
    # Optional: Log memory usage for debugging
    if hasattr(psutil, 'Process'):
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        except:
            pass  # Ignore if psutil is not available

def process_ocr_pipeline(
    img: np.ndarray,
    segmentation_service,
    ocr_service,
    id_detection_service,
    classification_service,
    source: str = "file"
) -> OCRResult:
    segmented_card = None
    request_id = str(uuid.uuid4())[:8]

    logger.info(f"Starting OCR pipeline for {source} input")
    logger.info(f"Image hash: {hash_image(img)}")
    logger.debug(f"Image dimensions: {img.shape}")
    try:

        if not validate_image_for_ocr(img):
            logger.warning("Image may not be suitable for OCR")


        # Step 2: Segmentation
        logger.info(f"[{request_id}] Step 1: Starting segmentation")

        segmented_card = segmentation_service.crop_id_card(img)
        if segmented_card is None:
            logger.warning(f"[{request_id}] Could not detect card during segmentation")
            return OCRResult(status=False, message="Could not detect card", body=None)

        logger.info(f"[{request_id}] Segmentation completed succesfully")

        # Step 1: Classification
        logger.info(f"[{request_id}] Step 2: Starting classification")

        classified_card_type = classification_service.classify_card(segmented_card)
        if classified_card_type is None:
            return OCRResult(status=False, message="No ID card detected", body=None)

        logger.info(f"Card classified as: {classified_card_type}")

        # Step 3: Field Detection
        logger.info(f"[{request_id}] Step 3: Starting field detection")
        detected_fields = id_detection_service.extract_fields(segmented_card, classified_card_type)
        # logger.info(f"[{request_id}] Detected fields: {detected_fields}")
        if not detected_fields:
            logger.warning(f"[{request_id}] No fields detected")
            return OCRResult(status=False, message="No fields detected", body=None)

        # Step 4: OCR
        logger.info(f"[{request_id}] Step 4: Starting OCR processing")
        extracted_fields = {}
        failed_fields = []
        for i, field in enumerate(detected_fields):
            field_name = field.get("field_name", "unknown")
            field_crop = field.get("crop")
            logger.debug(f"[{request_id}] Processing field {i+1}/{len(detected_fields)}: {field_name}")
            if field_crop is None or field_crop.size == 0:
                logger.warning(f"[{request_id}] Field {field_name} has no crop data")
                continue

            raw_text = ocr_service.ocr_text(field_crop)
            logger.info(f"[{request_id}] RAW OCR for field '{field_name}': '{raw_text}'") # Add this line

            if raw_text and raw_text.strip():
                cleaned = clean_text(raw_text, field_name, classified_card_type)
                if cleaned:
                    extracted_fields[field_name] = cleaned
                    logger.debug(f"[{request_id}] Field {field_name}: '{cleaned}'")
                else:
                    logger.warning(f"[{request_id}] Field {field_name}: cleaning resulted in empty text")
            else:
                failed_fields.append(field_name)
                logger.warning(f"[{request_id}] Field {field_name}: OCR returned empty text")
        
        logger.info(f"[{request_id}] OCR completed. Extracted {len(extracted_fields)} fields, failed {len(failed_fields)} fields")

        # Step 5: Voter card postprocessing
        if classified_card_type == "voter":
            logger.info(f"[{request_id}] Step 5: Voter card postprocessing")

            try:
                extracted_fields = filter_voter_fields(extracted_fields)
                logger.info(f"[{request_id}] Voter fields filtered successfully")

            except HTTPException as e:
                logger.error(f"[{request_id}] Voter card postprocessing failed: {e.detail}")

                return OCRResult(status=False, message="Please upload front side of voter card", body=None)

        # Step 6: Main ID validation
        logger.info(f"[{request_id}] Step 6: Main ID validation")

        if not validate_main_id_field(extracted_fields, classified_card_type):
            logger.warning(f"[{request_id}] Main ID field validation failed")

            return OCRResult(status=False, message=f"Missing required ID Number for {classified_card_type}", body=None)
        
        logger.info(f"[{request_id}] Pipeline completed successfully")
        normalized_fields  = normalize_fields(classified_card_type, extracted_fields)
        card_type_name = classified_card_type.lower()
        card_type_id = settings.CARD_TYPE_IDS.get(card_type_name, 0)  # fallback to 0 = unknown
        return OCRResult(
            status=True,
            message=f"OCR successful for {classified_card_type}",
            body=OCRResultBody(
                card_type_id=card_type_id,
                card_type=classified_card_type,
                name=normalized_fields.get("name"),
                id_number=normalized_fields.get("id_number"),
                dob=normalized_fields.get("dob"),
                address=normalized_fields.get("address"),
                gender=normalized_fields.get("gender"),
                expiry_date=normalized_fields.get("expiry_date"),
                coname=normalized_fields.get("coname"),
            )
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        logger.error(traceback.format_exc())
        return OCRResult(status=False, message=f"Error: {str(e)}", body=None)

    finally:
        cleanup_memory()
        if segmented_card is not None:
            del segmented_card
        logger.debug(f"[{request_id}] Pipeline cleanup completed")



@router.post("/", response_model=OCRResult, response_model_exclude_unset=True)
async def run_ocr(file: UploadFile = File(...)):
    """
    OCR Pipeline with File Upload: Classification → Segmentation → Field Detection → OCR
    """
    try:
        from app.services.segmentation import SegmentationService
        # from app.services.easyocr_service import EasyOCRService
        from app.services.paddleocr_service import PaddleOCRService
        from app.services.id_detection import IDDetectionService
        from app.services.classification import ClassificationService

        # GPU check
        # check_gpu_availability()

        # Load image
        img = await read_imagefile(file)
        logger.info(f"Image loaded successfully from file: {file.filename}")

        # Create service instances (stateless, per request)
        segmentation_service = SegmentationService()
        # ocr_service = EasyOCRService()
        ocr_service = PaddleOCRService()
        id_detection_service = IDDetectionService()
        classification_service = ClassificationService()

        # Run OCR pipeline
        return process_ocr_pipeline(
            img,
            segmentation_service,
            ocr_service,
            id_detection_service,
            classification_service,
            source="file"
        )

    except HTTPException:
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
    OCR Pipeline with Base64 Input: Classification → Segmentation → Field Detection → OCR
    """
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received base64 image request")

    
    base64_data_len = len(request.image_data)
    # truncated_base64 = request.image_data[:100] + "..." if base64_data_len > 100 else request.image_data
    logger.debug(f"[{request_id}] Base64 data length: {base64_data_len}")

    img = None

    try:
        base64_hash = hashlib.md5(request.image_data.encode()).hexdigest()
        logger.info(f"[{request_id}] Base64 input hash: {base64_hash}")

        # logger.info(f"[{request_id}] Starting base64 OCR request")
        # logger.info(f"[{request_id}] Base64 data length: {len(request.image_data)}")

        # Check for GPU if required
        # check_gpu_availability()

        # Decode base64 image
        img = decode_base64_image(request.image_data)
        logger.info(f"[{request_id}] Image decoded successfully")
        # logger.info(f"[{request_id}] Full Base64 data: {request.image_data}") 


        # Create fresh instances of all services
        from app.services.segmentation import SegmentationService
        # from app.services.easyocr_service import EasyOCRService
        from app.services.paddleocr_service import PaddleOCRService
        from app.services.id_detection import IDDetectionService
        from app.services.classification import ClassificationService

        segmentation_service = SegmentationService()
        # ocr_service = EasyOCRService()
        ocr_service = PaddleOCRService()
        id_detection_service = IDDetectionService()
        classification_service = ClassificationService()

        # Run the OCR pipeline
        result = process_ocr_pipeline(
            img,
            segmentation_service,
            ocr_service,
            id_detection_service,
            classification_service,
            source="base64"
        )

        logger.info(f"[{request_id}] OCR request completed successfully")
        logger.info(f"[{request_id}] Decoded image hash: {hash_image(img)}")

        return result

    except HTTPException:
        logger.error(f"[{request_id}] HTTPException occurred")
        raise

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in base64 OCR: {e}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during base64 processing"
        )

    finally:
        if img is not None:
            del img
        cleanup_memory()

@router.get("/health")
def health_check():
    try:
        _ = SegmentationService()
        # _ = EasyOCRService()
        _ = PaddleOCRService()
        _ = IDDetectionService()
        _ = ClassificationService()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))