from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Optional, Union, ClassVar
import os
import logging
from pathlib import Path
import sys
import logging.handlers # Import for RotatingFileHandler

from dotenv import load_dotenv
load_dotenv()

# --- Helper function to get the correct base path for bundled resources ---
def get_resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a resource, handling PyInstaller's temporary directory.
    """
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        # sys._MEIPASS is the path to the temporary folder where the bundle is extracted.
        base_path = sys._MEIPASS
    else:
        # Running as a normal Python script (during development)
        # Assume the script is run from the project root (D:\ocr-text\)
        base_path = os.getcwd()
    
    # Join the base path with the relative path to the resource
    # For models, relative_path will be like "models/fine_tuned_rotation_model.pth"
    return os.path.join(base_path, relative_path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        extra='allow',
        env_file_encoding = "utf-8",
        case_sensitive = True
    )

    # API Settings
    API_TITLE: str = "ID Card OCR & Field Extraction API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Classification → Segmentation → Field Detection → OCR Pipeline"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Settings
    CORS_ORIGINS: str = "*"
    CORS_ORIGINS_LIST: List[str] = []

    # Model Paths (These will now store the *absolute* paths after __init__)
    # The values from .env or defaults will be relative, and then converted.
    MODEL_DIR_RELATIVE: str = os.getenv("MODEL_DIR", "models") # Keep original for reference

    CLASSIFICATION_MODEL_PATH_RELATIVE: str = os.getenv("CLASSIFICATION_MODEL_PATH", "models/fine_tuned_rotation_model.pth")
    SEGMENTATION_MODEL_PATH_RELATIVE: str = os.getenv("SEGMENTATION_MODEL_PATH", "models/segmentation_model.pt")
    
    FIELD_DETECTION_MODELS_RELATIVE: Dict[str, str] = {
        "aadhar": os.getenv("AADHAR_MODEL_PATH", "models/best_aadhar_ocr_model.pt"),
        "pan": os.getenv("PAN_MODEL_PATH", "models/pan_model.pt"),
        "voter": os.getenv("VOTER_MODEL_PATH", "models/finetuned_obb_model_voter.pt"),
        "driving": os.getenv("DRIVING_MODEL_PATH", "models/driving.pt"),
    }

    # These will be the actual absolute paths used by services
    CLASSIFICATION_MODEL_PATH: str = ""
    SEGMENTATION_MODEL_PATH: str = ""
    FIELD_DETECTION_MODELS: Dict[str, str] = {}
    PADDLEOCR_WHLS_PATH: str = ""

    # ... (rest of your existing settings like CONFIDENCE_THRESHOLD, FIELD_LABELS, OCR_LANGUAGES, etc.) ...
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.8
    DEFAULT_CLASSIFICATION_CLASSES: List[str] = ["aadhar", "pan", "voter", "driving"]
    DEFAULT_CLASSIFICATION_MODEL_BACKBONE: str = "resnet50"

    SEGMENTATION_USE_MASK: bool = True
    SEGMENTATION_CONFIDENCE_THRESHOLD: float = 0.5

    FIELD_DETECTION_CONFIDENCE_THRESHOLD: float = 0.5

    FIELD_LABELS: Dict[str, List[str]] = {
        "aadhar": ["AADHAR_NUMBER", "ADDRESS", "DATE_OF_BIRTH","GENDER", "NAME"]
    }
    FIELD_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "voter": {
            "name": ["name", "Name", "NAME", "full_name", "voter_name"],
            "id_number": ["id_number", "number", "voter_id", "epic_no"],
            "dob": ["dob", "date_of_birth", "birth_date"]
        },
        "pan": {
            "name": ["Name"],
            "id_number": ["PAN Number"],
            "dob":[ "DOB"],
            "coname": ["Father Name"],
        },
        "aadhar": {
            "name": ["NAME"],
            "id_number": ["AADHAR_NUMBER"],
            "address": ["ADDRESS"],
            "dob": ["DATE_OF_BIRTH"],
            "gender": ["GENDER"],
        },
        "driving": {
            "name": ["Name"],
            "id_number":  ["Number"],
            "dob": ["dob"],
            "coname": ["father_name"],
            "expiry_date": ["expiry_date"],
        }
    }

    OCR_LANGUAGES: str = "en"
    OCR_USE_GPU: bool = False
    OCR_MIN_CONFIDENCE_THRESHOLD: float = 0.5

    CARD_TYPE_IDS: Dict[str, int] = {
        "aadhar": 1,
        "pan": 2,
        "voter": 3,
        "driving": 4
    }

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/api.log")

    def model_post_init(self, handler=None) -> None:
        # Process CORS_ORIGINS string into a list
        if isinstance(self.CORS_ORIGINS, str):
            self.CORS_ORIGINS_LIST = [origin.strip() for origin in self.CORS_ORIGINS.split(',') if origin.strip()]
        else:
            self.CORS_ORIGINS_LIST = ["*"]

        # Convert relative model paths to absolute paths using the resource helper
        self.CLASSIFICATION_MODEL_PATH = get_resource_path(self.CLASSIFICATION_MODEL_PATH_RELATIVE)
        self.SEGMENTATION_MODEL_PATH = get_resource_path(self.SEGMENTATION_MODEL_PATH_RELATIVE)

        self.FIELD_DETECTION_MODELS = {
            card_type: get_resource_path(path)
            for card_type, path in self.FIELD_DETECTION_MODELS_RELATIVE.items()
        }

        # Special handling for PaddleOCR's whl models if they are in a specific subfolder
        # Assuming you copied them to 'models/paddleocr_whl'
        self.PADDLEOCR_WHLS_PATH = get_resource_path(os.path.join(self.MODEL_DIR_RELATIVE, 'paddleocr_whl'))


    def validate_model_paths(self) -> bool:
        """Validate if all configured model paths exist."""
        all_paths_exist = True

        # Classification model
        if self.CLASSIFICATION_MODEL_PATH and not os.path.exists(self.CLASSIFICATION_MODEL_PATH):
            logger.warning(f"Classification model not found: {self.CLASSIFICATION_MODEL_PATH}. Ensure it's present if ClassificationService uses a local model, or that the Torch microservice handles classification fully.")
            all_paths_exist = False

        # Segmentation model
        if self.SEGMENTATION_MODEL_PATH and not os.path.exists(self.SEGMENTATION_MODEL_PATH):
            logger.warning(f"Missing model file: {self.SEGMENTATION_MODEL_PATH}")
            all_paths_exist = False

        # Field detection models
        for card_type, path in self.FIELD_DETECTION_MODELS.items():
            if path and not os.path.exists(path):
                logger.warning(f"Missing model file: {path}")
                all_paths_exist = False
        
        # Check PaddleOCR whl models base path
        if not os.path.exists(self.PADDLEOCR_WHLS_PATH):
            logger.warning(f"PaddleOCR models base directory not found: {self.PADDLEOCR_WHLS_PATH}. Ensure 'paddleocr_whl' is in your 'models' folder.")
            all_paths_exist = False

        return all_paths_exist

def setup_logging(settings: Settings):
    # Ensure logs directory exists
    log_dir = os.path.dirname(settings.LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Configure basic logging for the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)

    # Clear existing handlers to prevent duplicate logs
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    # Add your custom app loggers
    app_loggers = [
        "app.main",
        "app.routers.ocr",
        "app.services.paddleocr_service",
        "app.services.segmentation",
        "app.services.utils",
        "app.services.classification",
        "app.services.id_detection",
        "app.config"
    ]

    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(settings.LOG_LEVEL)
        logger.propagate = True

    # Set specific logger levels for noisy third-party libraries
    third_party_loggers = {
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "PIL": logging.WARNING,
        "httpx": logging.WARNING,
        "ultralytics": logging.WARNING,
        "paddleocr": logging.WARNING,
        "ppocr": logging.WARNING,
        "torch": logging.WARNING,
        "uvicorn": logging.INFO,
        "uvicorn.access": logging.INFO,
        "fastapi": logging.INFO,
    }

    for logger_name, level in third_party_loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = True

    # Test logging setup
    test_logger = logging.getLogger("app.config")
    test_logger.info("Logging configuration completed successfully")

    return root_logger

# Initialize settings
settings = Settings()

# Set up logging
setup_logging(settings)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info("Configuration loaded successfully")

# Validate model paths (optional - log warnings but don't fail)
if not settings.validate_model_paths():
    logger.warning("Some model files are missing. Ensure all model files are in place before running the service.")
