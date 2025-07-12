# app/main.py
import uvicorn
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.routers import ocr
import logging
import os

# Import config first to set up logging
from app.config import settings

# Now import other modules
from app.routers.ocr import router as ocr_router

os.environ['FLAGS_log_level'] = os.getenv("PADDLE_LOG_LEVEL","2")

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting up ID Card OCR API")
    try:
        # Validate GPU if required
        # if settings.OCR_USE_GPU:
        #     import torch
        #     if not torch.cuda.is_available():
        #         logger.error("GPU is required but not available")
        #         raise RuntimeError("GPU is required but not available")
        #     logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        
        # Validate model paths
        if not settings.validate_model_paths():
            logger.warning("Some model files are missing, but continuing startup")
        
        logger.info("Startup completed successfully")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down ID Card OCR API")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include OCR router
app.include_router(
    ocr_router,
    prefix="/ocr",
    tags=["ocr"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.get("/", tags=["root"])
async def read_root():
    """Root endpoint - API information"""
    logger.info("Root endpoint accessed")
    return {
        "message": "ID Card OCR API is running",
        "version": settings.API_VERSION,
        "pipeline": "Classification → Segmentation → Field Detection → OCR",
        "docs": "/docs",
        "status": "healthy"
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    
    # Check GPU if required
    gpu_status = "not_required"
    if settings.OCR_USE_GPU:
        import torch
        gpu_status = "available" if torch.cuda.is_available() else "unavailable"
    
    return {
        "status": "healthy",
        "service": "ID Card OCR API",
        "version": settings.API_VERSION,
        "gpu_status": gpu_status,
        "log_level": settings.LOG_LEVEL
    }

if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {settings.HOST}:{settings.PORT}")
    try:
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=False,  # Set to True for development
            log_level=settings.LOG_LEVEL.lower()
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)