# Trademark Comparison API

An AI-powered trademark image similarity search and PDF processing system built with FastAPI. This system enables efficient trademark comparison, similarity search, and automated extraction of trademark images from PDF documents using deep learning embeddings and vector similarity search.

## Features

- **Image Similarity Search**: Upload trademark images and find similar trademarks using CLIP-based embeddings
- **PDF Processing**: Extract trademark images from PDF documents with automatic image detection
- **Vector Database**: Fast similarity search powered by Qdrant vector database
- **Batch Processing**: Process multiple PDFs and images in batch operations
- **RESTful API**: Comprehensive FastAPI-based REST API with interactive documentation
- **Frontend Interface**: Modern Next.js web application for user interaction
- **Authentication**: Secure session-based authentication system

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **PyTorch & Transformers**: Deep learning models (CLIP)
- **Qdrant**: Vector database for similarity search
- **Pillow & OpenCV**: Image processing
- **PyMuPDF**: PDF processing and image extraction

### Frontend
- **Next.js 16**: React framework
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Modern styling

## Prerequisites

- Python 3.8+ (Python 3.13 recommended)
- Node.js 18+ and npm
- Qdrant vector database (local or cloud instance)
- CUDA-capable GPU (optional, for faster processing)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Jitam-Bharadwaj/trademark-comparison-backend.git
cd Trademark-pyBackend
```

### 2. Backend Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Frontend Setup

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
cd ..
```

### 4. Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# Model Configuration
EMBEDDING_MODEL=openai/clip-vit-base-patch32
DEVICE=cuda  # or 'cpu' if no GPU available

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=trademarks
QDRANT_API_KEY=  # Optional, for cloud instances

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE_MB=100

# Database Configuration (if using MySQL)
DATABASE_USERNAME=root
DATABASE_PASSWORD=
DATABASE_HOSTNAME=localhost
DATABASE_PORT=3306
DATABASE_NAME=devehost_trademark

# Processing Configuration
PDF_DPI=300
SIMILARITY_THRESHOLD=0.5
LOG_LEVEL=INFO
```

### 5. Start Qdrant Database

**Option A: Local Qdrant (Docker)**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Cloud Qdrant**

Configure `QDRANT_HOST` and `QDRANT_API_KEY` in your `.env` file.

## Running the Application

### Start the Backend API Server

```bash
python scripts/run_api_server.py
```

Or directly with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Start the Frontend Development Server

In a separate terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Access API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints Overview

### Authentication
- `POST /auth/signup` - User registration
- `POST /auth/login` - User login
- `POST /auth/logout` - User logout

### Image Search
- `POST /search` - Search for similar trademarks
- `POST /batch-search` - Batch image search
- `GET /trademark/{trademark_id}` - Get trademark details
- `GET /image/{trademark_id}` - Retrieve trademark image

### PDF Processing
- `POST /extract-pdf` - Extract images from PDF
- `POST /extract-pdf-batch` - Batch PDF processing
- `POST /process-pdf-and-index` - Extract and index in one step
- `POST /index-extracted-images` - Index extracted images to vector DB

### Database Management
- `GET /stats` - Get database statistics
- `DELETE /trademark/{trademark_id}` - Delete trademark

### Health & Status
- `GET /health` - Health check endpoint

## Project Structure

```
Trademark-pyBackend/
├── api/                 # API routes and data storage
├── auth/                # Authentication modules
├── database/            # Vector database and connection management
├── frontend/            # Next.js frontend application
├── indexing/            # Batch indexing utilities
├── models/              # ML models and embedding generators
├── pdf_processing/      # PDF extraction and processing
├── preprocessing/       # Image preprocessing utilities
├── scripts/             # Utility scripts
├── utils/               # Logging and helper utilities
├── config.py            # Configuration management
├── main.py              # FastAPI application entry point
└── requirements.txt     # Python dependencies
```

## Development

### Running Tests

```bash
pytest tests/
```

### Batch Indexing

Index multiple trademark images:

```bash
python scripts/run_batch_indexing.py
```

## License

[Specify your license here]

## Support

For issues and questions, please open an issue in the repository.

