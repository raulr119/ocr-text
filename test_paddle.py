# test_paddleocr.py
import cv2
import numpy as np
from app.services.paddleocr_service import PaddleOCRService
import logging

logging.basicConfig(level=logging.DEBUG) # Set to DEBUG to see all logs

# Path to a sample image
image_path = r"D:\images\driving.png" # <--- IMPORTANT: Change this

try:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
    else:
        print(f"Image loaded successfully with shape: {img.shape}")
        paddle_ocr_service = PaddleOCRService()
        extracted_text = paddle_ocr_service.ocr_text(img)
        print(f"Extracted Text: {extracted_text}")
except Exception as e:
    print(f"An error occurred: {e}")