"""
Script to run the API server
Usage: python run_api_server.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import uvicorn
from config import Config

if __name__ == "__main__":
    print("Starting Trademark Comparison API Server...")
    print(f"Host: {Config.API_HOST}")
    print(f"Port: {Config.API_PORT}")
    print(f"Model: {Config.EMBEDDING_MODEL}")
    print(f"Device: {Config.DEVICE}")
    
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False,
        log_level="info"
    )
