# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Optional, Union, ClassVar
import os
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    # This configuration tells Pydantic to load environment variables from a .env file
    # and ignore any extra fields in the .env file that are not defined here.
    model_config = SettingsConfigDict(
        env_file='.env', 
        extra='ignore',
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
    CORS_ORIGINS: str = "*"  # This will be processed in model_post_init
    CORS_ORIGINS_LIST: List[str] = []  # Will be populated in model_post_init
    
    # Model Paths
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    
    # Classification Model
    CLASSIFICATION_MODEL_PATH: str = ""
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.8
    
    # Segmentation Model
    SEGMENTATION_MODEL_PATH: str = ""
    SEGMENTATION_CONFIDENCE_THRESHOLD: float = 0.6
    SEGMENTATION_USE_MASK: bool = True
    
    # Field Detection Models - will be built in model_post_init
    AADHAR_MODEL_PATH: str = ""
    PAN_MODEL_PATH: str = ""
    VOTER_MODEL_PATH: str = ""
    DRIVING_MODEL_PATH: str = ""
    FIELD_DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    
    # # OCR Settings
    OCR_LANGUAGES: str = "en"  # This will be processed in model_post_init
    OCR_USE_GPU: bool = False
    OCR_PREPROCESS: bool = False
    OCR_UPSCALE_FACTOR: int = 2
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = ""  # Empty string means console only
    
    # Processed fields (will be populated in model_post_init)
    OCR_LANGUAGES_LIST: List[str] = []
    CORS_ORIGINS_LIST: List[str] = []
    FIELD_DETECTION_MODELS: Dict[str, str] = {}

    DEFAULT_CLASSIFICATION_CLASSES: List[str] = ["aadhar", "driving", "pan", "voter"]
    DEFAULT_CLASSIFICATION_MODEL_BACKBONE: str = "resnet50"
    
    FIELD_LABELS : ClassVar[Dict[str, List[str]]] = {
    "aadhar": ["AADHAR_NUMBER", "DATE_OF_BIRTH", "GENDER", "NAME", "ADDRESS"]
    }

    def model_post_init(self, __context) -> None:
        """Process string fields into lists/dicts after initialization"""
        # # Set default paths if not provided
        # if not self.CLASSIFICATION_MODEL_PATH:
        #     self.CLASSIFICATION_MODEL_PATH = os.path.join(self.MODEL_DIR, "classification_model.pth")
        
        # if not self.SEGMENTATION_MODEL_PATH:
        #     self.SEGMENTATION_MODEL_PATH = os.path.join(self.MODEL_DIR, "segmentation_model.pt")
        
        # if not self.AADHAR_MODEL_PATH:
        #     self.AADHAR_MODEL_PATH = os.path.join(self.MODEL_DIR, "aadhar_model.pt")
        
        # if not self.PAN_MODEL_PATH:
        #     self.PAN_MODEL_PATH = os.path.join(self.MODEL_DIR, "pan_model.pt")
        
        # if not self.VOTER_MODEL_PATH:
        #     self.VOTER_MODEL_PATH = os.path.join(self.MODEL_DIR, "voter_model.pt")
        
        # if not self.DRIVING_MODEL_PATH:
        #     self.DRIVING_MODEL_PATH = os.path.join(self.MODEL_DIR, "driving_model.pt")
        
        # Process OCR_LANGUAGES
        # self.OCR_LANGUAGES_LIST = [lang.strip() for lang in self.OCR_LANGUAGES.split(",") if lang.strip()]
        
        # Process CORS_ORIGINS
        if self.CORS_ORIGINS == "*":
            self.CORS_ORIGINS_LIST = ["*"]
        else:
            self.CORS_ORIGINS_LIST = [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        
        # Build FIELD_DETECTION_MODELS
        self.FIELD_DETECTION_MODELS = {
            "aadhar": self.AADHAR_MODEL_PATH,
            "pan": self.PAN_MODEL_PATH,
            "voter": self.VOTER_MODEL_PATH,
            "driving": self.DRIVING_MODEL_PATH,
        }

        # Ensure the base model directory exists
        Path(self.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        
        # Convert OCR_LANGUAGES to a list if it's a string
        # if isinstance(self.OCR_LANGUAGES, str):
        #     self.OCR_LANGUAGES = [lang.strip() for lang in self.OCR_LANGUAGES.split(',') if lang.strip()]
    
    # def validate_model_paths(self) -> bool:
    #     """Validate that all model paths exist"""
    #     paths_to_check = [
    #         self.CLASSIFICATION_MODEL_PATH,
    #         self.SEGMENTATION_MODEL_PATH,
    #     ]
    #     paths_to_check.extend(self.FIELD_DETECTION_MODELS.values())
        
    #     missing_paths = []
    #     for path in paths_to_check:
    #         if not os.path.exists(path):
    #             missing_paths.append(path)
        
    #     if missing_paths:
    #         logging.warning(f"Missing model files: {missing_paths}")
    #         return False
    #     return True

    def validate_model_paths(self) -> bool:
        """Checks if all configured model files exist."""
        all_models_exist = True
        # List of model paths that are required for the main application's local services
        # Note: Paths for microservices (Paddle/Torch) are not validated here, as their models are internal to their services.
        required_models = [
            self.SEGMENTATION_MODEL_PATH,
        ] + list(self.FIELD_DETECTION_MODELS.values()) # Field detection models are local

        # Classification model is potentially local if not using microservice
        # Keeping this check here for completeness, though it might be redundant with microservice setup.
        if Path(self.CLASSIFICATION_MODEL_PATH).is_file():
            # If a local classification model is expected, add it to required_models for validation
            # This depends on whether ClassificationService relies on a local file or is purely via Torch microservice.
            # Assuming ClassificationService still has a local model for its `classify_card` method.
            required_models.append(self.CLASSIFICATION_MODEL_PATH)
        else:
            logging.warning(f"Classification model file not found: {self.CLASSIFICATION_MODEL_PATH}. "
                            "Ensure it's present if ClassificationService uses a local model, or that "
                            "the Torch microservice handles classification fully.")


        for path in required_models:
            if not Path(path).is_file():
                logging.warning(f"Missing model file: {path}")
                all_models_exist = False
        return all_models_exist
def setup_logging(settings: Settings):
    """Configure logging based on settings"""
    # Create formatter
    # formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Configure root logger
    root_logger = logging.getLogger()
    # root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    #Set root logger level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(),logging.INFO)
    root_logger.setLevel(log_level)

    #Create formatter
    formatter = logging.Formatter(settings.LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler (if LOG_FILE is specified)
    if settings.LOG_FILE:
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(settings.LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(settings.LOG_FILE)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)
            
            logging.info(f"Logging to file: {settings.LOG_FILE}")
        except Exception as e:
            logging.error(f"Failed to create file handler: {e}")
    
    # Configure specific loggers with explicit levels
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

    # Set our app loggers to the configured level
    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        # Don't propagate to avoid duplicate messages
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
else:
    logger.info("All model paths validated successfully")