import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration for the trademark system"""
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "512"))
    DEVICE = os.getenv("DEVICE", "cuda")
    
    # Vector Database
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "trademarks")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    
    # Paths
    BASE_DIR = Path(__file__).parent
    TRADEMARK_IMAGES_PATH = Path(os.getenv("TRADEMARK_IMAGES_PATH", "./data/trademarks"))
    TRADEMARK_PDFS_PATH = Path(os.getenv("TRADEMARK_PDFS_PATH", "./data/pdfs"))
    EXTRACTED_IMAGES_PATH = Path(os.getenv("EXTRACTED_IMAGES_PATH", "./data/extracted"))
    UPLOAD_PATH = Path(os.getenv("UPLOAD_PATH", "./data/uploads"))
    LOG_PATH = Path(os.getenv("LOG_PATH", "./logs"))
    
    # PDF Processing
    PDF_DPI = int(os.getenv("PDF_DPI", "300"))
    MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", "100"))  # Minimum width/height in pixels
    MAX_IMAGES_PER_PDF = int(os.getenv("MAX_IMAGES_PER_PDF", "50"))  # Safety limit
    
    # Processing
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    
    # Monitoring
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Database Configuration
    DATABASE_USERNAME = os.getenv("DATABASE_USERNAME", "root")
    DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "")
    DATABASE_HOSTNAME = os.getenv("DATABASE_HOSTNAME", "localhost")
    DATABASE_PORT = int(os.getenv("DATABASE_PORT", "3306"))
    DATABASE_NAME = os.getenv("DATABASE_NAME", "devehost_trademark")
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.TRADEMARK_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
        cls.TRADEMARK_PDFS_PATH.mkdir(parents=True, exist_ok=True)
        cls.EXTRACTED_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
        cls.LOG_PATH.mkdir(parents=True, exist_ok=True)