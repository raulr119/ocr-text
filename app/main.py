# app/main.py

import uvicorn
import sys
import os
import logging
import traceback # Import traceback for detailed error logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

if getattr(sys, 'frozen', False):
    # This allows 'app.config' and 'app.routers' to be imported correctly.
    sys.path.insert(0, sys._MEIPASS)
    

# --- IMPORTANT: All 'app' imports MUST come AFTER the sys.path modification ---
from app.config import Settings, setup_logging
from app.routers.ocr import router as ocr_router

os.environ['FLAGS_log_level'] = os.getenv("PADDLE_LOG_LEVEL", "2")

settings = Settings()
setup_logging(settings)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting up ID Card OCR API")
    try:
        # Validate model paths
        if not settings.validate_model_paths():
            logger.warning("Some model files are missing, but continuing startup")

        logger.info("Startup completed successfully")
        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(traceback.format_exc()) # Log full traceback
        raise RuntimeError(f"Application startup failed: {e}")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(ocr_router, prefix="/ocr", tags=["ocr"])

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled Exception: {exc}")
    logger.error(traceback.format_exc()) # Log full traceback
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
    import argparse
    parser = argparse.ArgumentParser(description="Run ID Card OCR API.")
    parser.add_argument('--host', type=str, default=settings.HOST,
                        help=f'Host address for the API (default: {settings.HOST}).')
    parser.add_argument('--port', type=int, default=settings.PORT,
                        help=f'Port number for the API (default: {settings.PORT}).')
    args = parser.parse_args()

    logger.info(f"Starting Uvicorn server on {args.host}:{args.port}") # Corrected args.HOST to args.host
    print(f"✅ ID Card OCR API running at http://{args.host}:{args.port}")  # Early feedback for .exe

    try:
        uvicorn.run(
            "app.main:app", # This string path is correct for Uvicorn
            host=args.host,
            port=args.port,
            reload=False, # Always False for bundled apps
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True,
            workers=4
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
