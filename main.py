from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import io
import logging
import re
from datetime import datetime
from pathlib import Path
from PIL import Image

# Import local modules
from config import Config
from preprocessing.image_processor import ImagePreprocessor
from models.embedding_model import EmbeddingGenerator
from database.vector_db import VectorDatabase
from utils.logger import setup_logger
from pdf_processing.pdf_extractor import PDFTrademarkExtractor
from api.auth_routes import router as auth_router
from utils.ocr_extractor import OCRTextExtractor
from utils.text_similarity import TextSimilarity
from database.application_queries import application_queries
from PIL.PngImagePlugin import PngImageFile

# Increase the MAX_TEXT_CHUNK limit (default is ~1MB, increase to 10MB)
Image.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 10MB

# Initialize FastAPI app
app = FastAPI(
    title="Trademark Comparison API",
    description="AI-powered trademark image similarity search and PDF processing system",
    version="1.0.0",
    tags_metadata=[
        {
            "name": "Health & Status",
            "description": "Health check and system status endpoints",
        },
        {
            "name": "Authentication",
            "description": "User authentication endpoints (signup, login, logout)",
        },
        {
            "name": "Image Search",
            "description": "Trademark similarity search using image uploads",
        },
        {
            "name": "PDF Processing",
            "description": "Extract and process trademarks from PDF documents",
        },
        {
            "name": "Database Management",
            "description": "Manage trademark database and view statistics",
        },
        {
            "name": "Debug & Testing",
            "description": "Debug endpoints for testing and troubleshooting",
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trademark-ai-comp-frontend.vercel.app",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (project root) to avoid cross-origin issues in dev
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent)), name="static")

# Simple routes to serve pages from same origin
@app.get("/", include_in_schema=False)
async def serve_login_page():
    return FileResponse(str(Path(__file__).parent / "login.html"))


@app.get("/app", include_in_schema=False)
async def serve_frontend_page():
    return FileResponse(str(Path(__file__).parent / "frontend.html"))

# Include authentication routes
app.include_router(auth_router)

# Global instances (will be initialized on startup)
preprocessor = None
embedding_generator = None
vector_db = None
pdf_extractor = None
ocr_extractor = None
text_similarity = None
logger = None

# Pydantic models
class SearchResult(BaseModel):
    trademark_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    vector_similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    text_similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    matched_mark: Optional[str] = None
    source: str = Field(..., description="Source of trademark: 'Self Database Trademark' or 'Indian Trademark Journal'")
    trademark_class: Optional[str] = Field(None, description="Trademark class")
    applicant_name: Optional[str] = Field(None, description="Applicant name")
    application_no: Optional[str] = Field(None, description="Application number")
    trademark_type: Optional[str] = Field(None, description="Type: 'text_only' or 'image_based'")
    has_image: Optional[bool] = Field(None, description="Whether this trademark has an associated image")
    metadata: dict

class SearchResponse(BaseModel):
    query_time_ms: float
    total_results: int
    extracted_text: Optional[str] = None
    results: List[SearchResult]

class TextSearchRequest(BaseModel):
    query_text: str = Field(..., description="Text query to search for similar trademarks")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold (0-1)")
    trademark_class: Optional[str] = Field(None, description="Filter by trademark class")

class TrademarkInfo(BaseModel):
    trademark_id: str
    metadata: dict

class DatabaseStats(BaseModel):
    total_trademarks: int
    vector_dimension: int
    distance_metric: str

class ExtractedImage(BaseModel):
    image_id: str
    width: int
    height: int
    page: int
    extraction_method: str
    metadata: dict

class PDFExtractionResult(BaseModel):
    pdf_filename: str
    total_images_extracted: int
    images: List[ExtractedImage]
    processing_time_ms: float

class PDFBatchResult(BaseModel):
    batch_id: str
    total_pdfs: int
    successful_pdfs: int
    failed_pdfs: int
    total_images_extracted: int
    processing_time_ms: float
    results: List[PDFExtractionResult]

# Authentication removed - all endpoints are now public

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global preprocessor, embedding_generator, vector_db, pdf_extractor, ocr_extractor, text_similarity, logger
    
    # Create directories
    Config.create_directories()
    
    # Setup logger
    logger = setup_logger(
        "trademark_api",
        Config.LOG_PATH / f"api_{datetime.now().strftime('%Y%m%d')}.log",
        level=getattr(logging, Config.LOG_LEVEL)
    )
    logger.info("Starting Trademark Comparison API")
    
    # Initialize components
    logger.info("Initializing image preprocessor...")
    preprocessor = ImagePreprocessor(target_size=(224, 224), remove_bg=False)
    
    logger.info("Initializing PDF extractor...")
    pdf_extractor = PDFTrademarkExtractor(
        dpi=Config.PDF_DPI,
        min_size=Config.MIN_IMAGE_SIZE,
        max_images=Config.MAX_IMAGES_PER_PDF
    )
    
    # Initialize OCR extractor if enabled
    if Config.ENABLE_OCR:
        try:
            logger.info(f"Initializing OCR extractor (method: {Config.OCR_METHOD})...")
            ocr_extractor = OCRTextExtractor(
                ocr_method=Config.OCR_METHOD,
                language=Config.OCR_LANGUAGE
            )
            logger.info("OCR extractor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize OCR extractor: {e}. OCR will be disabled.")
            ocr_extractor = None
    else:
        logger.info("OCR is disabled in configuration")
        ocr_extractor = None
    
    # Initialize text similarity calculator
    logger.info("Initializing text similarity calculator...")
    text_similarity = TextSimilarity(
        use_levenshtein=True,
        use_phonetic=True,
        use_fuzzywuzzy=True
    )
    logger.info("Text similarity calculator initialized")
    
    # Pre-load marks from database if caching is enabled
    if Config.CACHE_MARKS:
        try:
            logger.info("Pre-loading marks from database...")
            marks = application_queries.get_all_marks(use_cache=True)
            logger.info(f"Pre-loaded {len(marks)} marks from database")
        except Exception as e:
            logger.warning(f"Failed to pre-load marks: {e}")
    
    logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
    embedding_generator = EmbeddingGenerator(
        model_name=Config.EMBEDDING_MODEL,
        device=Config.DEVICE
    )
    
    logger.info("Connecting to vector database...")
    try:
        # Determine connection parameters based on API key
        if Config.QDRANT_API_KEY:
            # Cloud instance - use URL instead of host:port
            qdrant_url = Config.QDRANT_HOST if Config.QDRANT_HOST.startswith('http') else f"https://{Config.QDRANT_HOST}"
            logger.info(f"Connecting to cloud Qdrant: {qdrant_url}")
            vector_db = VectorDatabase(
                host=qdrant_url,
                port=Config.QDRANT_PORT,
                collection_name=Config.QDRANT_COLLECTION_NAME,
                embedding_dim=embedding_generator.get_embedding_dimension(),
                api_key=Config.QDRANT_API_KEY
            )
        else:
            # Local instance
            logger.info(f"Connecting to local Qdrant: {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
            vector_db = VectorDatabase(
                host=Config.QDRANT_HOST,
                port=Config.QDRANT_PORT,
                collection_name=Config.QDRANT_COLLECTION_NAME,
                embedding_dim=embedding_generator.get_embedding_dimension(),
                api_key=None
            )
        logger.info("Successfully connected to vector database")
    except Exception as e:
        logger.error(f"Failed to connect to vector database: {e}")
        logger.warning("API will start without database connection. Some endpoints may not work.")
        vector_db = None
    
    logger.info("API startup complete")

# Health check endpoint
@app.get("/health", tags=["Health & Status"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": Config.EMBEDDING_MODEL,
        "device": Config.DEVICE,
        "database_connected": vector_db is not None
    }

# Search endpoint
@app.post("/search", response_model=SearchResponse, tags=["Image Search"])
async def search_similar_trademarks(
    file: UploadFile = File(...),
    top_k: int = 10,
    threshold: Optional[float] = None,
    trademark_class: Optional[str] = None
):
    """
    Search for similar trademarks using hybrid vector + text similarity
    
    Args:
        file: Uploaded trademark image
        top_k: Number of results to return
        threshold: Minimum similarity threshold (0-1)
        trademark_class: Filter by trademark class
        
    Returns:
        Search results with combined similarity scores
    """
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not available. Please ensure Qdrant is running.")
    
    import time
    start_time = time.time()
    
    try:
        # Validate file size
        contents = await file.read()
        if len(contents) > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {Config.MAX_FILE_SIZE_MB}MB"
            )
        
        # Load image
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        preprocessed = preprocessor.preprocess_pil(image)
        if preprocessed is None:
            raise HTTPException(status_code=400, detail="Failed to preprocess image")
        
        # Generate embedding for vector search
        embedding = embedding_generator.generate_embedding(preprocessed)
        
        # Extract text from image using OCR (if enabled)
        extracted_text = None
        if Config.ENABLE_OCR and ocr_extractor and ocr_extractor.is_available():
            try:
                extracted_text = ocr_extractor.extract_text(image)
                # Only use extracted text if it's meaningful (at least 2 characters)
                if extracted_text and len(extracted_text.strip()) >= 2:
                    logger.info(f"Extracted text from image: '{extracted_text[:50]}...' (truncated)")
                else:
                    logger.info("OCR extracted text is too short or empty, skipping text similarity")
                    extracted_text = None
            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")
                extracted_text = None
        
        # Vector similarity search
        filter_dict = {}
        if trademark_class:
            filter_dict['trademark_class'] = trademark_class
        
        search_threshold = threshold or Config.SIMILARITY_THRESHOLD
        logger.info(f"Vector search with threshold: {search_threshold}")
        
        vector_results = vector_db.search_similar(
            query_embedding=embedding,
            top_k=top_k * 2,  # Get more results for hybrid scoring
            score_threshold=search_threshold,
            filter_dict=filter_dict if filter_dict else None
        )
        
        logger.info(f"Vector search found {len(vector_results)} results")
        
        # Also search with text embedding if we have extracted text (for text-only trademarks)
        text_vector_results = []
        if extracted_text and embedding_generator:
            try:
                # Generate text embedding from extracted text
                text_embedding = embedding_generator.generate_text_embedding(extracted_text)
                logger.info(f"Generated text embedding for extracted text: '{extracted_text}'")
                
                # Search vector database with text embedding (this will find text-only trademarks)
                text_vector_results = vector_db.search_similar(
                    query_embedding=text_embedding,
                    top_k=top_k * 2,
                    score_threshold=search_threshold * 0.8,  # Slightly lower threshold for text-to-text matching
                    filter_dict=filter_dict if filter_dict else None
                )
                
                logger.info(f"Text embedding search found {len(text_vector_results)} results")
                
                # Add text vector results to vector_results, avoiding duplicates
                existing_ids = {r['trademark_id'] for r in vector_results}
                for text_result in text_vector_results:
                    if text_result['trademark_id'] not in existing_ids:
                        vector_results.append(text_result)
                        logger.info(f"Added text-only trademark from text embedding search: {text_result.get('metadata', {}).get('mark', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Text embedding search error: {e}", exc_info=True)
        
        # Text similarity search (if OCR extracted text) - for MySQL database marks
        text_results_map = {}
        mark_text_to_app_id = {}  # Map mark text to application_id for lookup
        
        if extracted_text and text_similarity:
            try:
                # Get all marks from database
                marks = application_queries.get_all_marks(use_cache=True)
                logger.info(f"Comparing extracted text '{extracted_text}' against {len(marks)} marks from database")
                
                # Create mark text to app_id mapping
                mark_texts = []
                for mark_data in marks:
                    if not mark_data.get('mark'):
                        continue
                    mark_text = mark_data['mark']
                    app_id = mark_data['application_id']
                    mark_texts.append(mark_text)
                    mark_text_to_app_id[mark_text] = app_id
                
                # Find best text matches using optimized method
                text_matches = text_similarity.find_best_matches(
                    query_text=extracted_text,
                    candidate_texts=mark_texts,
                    top_k=top_k * 2,
                    threshold=Config.TEXT_SIMILARITY_THRESHOLD
                )
                
                # Create map of application_id -> text similarity score
                for matched_text, text_score in text_matches:
                    app_id = mark_text_to_app_id.get(matched_text)
                    if app_id:
                        text_results_map[app_id] = {
                            'score': text_score,
                            'mark': matched_text
                        }
                
                logger.info(f"Text similarity found {len(text_results_map)} matches above threshold")
                
                # Debug: Log some sample marks for troubleshooting
                if len(text_results_map) == 0 and len(marks) > 0:
                    logger.debug(f"Sample marks from database (first 5): {[m['mark'] for m in marks[:5]]}")
                    logger.debug(f"Extracted text: '{extracted_text}' (normalized: '{text_similarity.normalize_text(extracted_text)}')")
                
                # If no matches found with fuzzy matching, try exact/partial match as fallback
                if len(text_results_map) == 0:
                    logger.info("No fuzzy matches found, trying exact/partial match fallback...")
                    normalized_extracted = text_similarity.normalize_text(extracted_text)
                    
                    for mark_data in marks:
                        if not mark_data.get('mark'):
                            continue
                        mark_text = mark_data['mark']
                        app_id = mark_data['application_id']
                        
                        # Normalize mark text
                        normalized_mark = text_similarity.normalize_text(mark_text)
                        
                        # Check for exact match (case-insensitive)
                        if normalized_extracted == normalized_mark:
                            text_results_map[app_id] = {
                                'score': 1.0,
                                'mark': mark_text
                            }
                            logger.info(f"Found exact match: '{mark_text}' (app_id: {app_id})")
                        # Check if extracted text is contained in mark or vice versa
                        elif normalized_extracted in normalized_mark or normalized_mark in normalized_extracted:
                            # Calculate similarity for partial matches
                            text_score = text_similarity.calculate_similarity(
                                extracted_text,
                                mark_text,
                                weights={
                                    'levenshtein': Config.LEVENSHTEIN_WEIGHT,
                                    'jaro_winkler': Config.JARO_WINKLER_WEIGHT,
                                    'token_sort': Config.TOKEN_SORT_WEIGHT,
                                    'phonetic': Config.PHONETIC_WEIGHT
                                }
                            )
                            if text_score >= 0.5:  # Lower threshold for partial matches
                                text_results_map[app_id] = {
                                    'score': text_score,
                                    'mark': mark_text
                                }
                                logger.info(f"Found partial match: '{mark_text}' (app_id: {app_id}, score: {text_score:.2f})")
                    
                    if text_results_map:
                        logger.info(f"Fallback search found {len(text_results_map)} matches")
                        
            except Exception as e:
                logger.error(f"Text similarity search error: {e}", exc_info=True)
                text_results_map = {}
        
        # Helper function to determine source of trademark
        # Create a set of all application IDs from text_results_map for quick lookup
        database_app_ids = set(text_results_map.keys()) if text_results_map else set()
        
        def determine_source(trademark_id: str, is_text_match: bool = False) -> str:
            """
            Determine the source of a trademark
            
            Args:
                trademark_id: Trademark ID
                is_text_match: Whether this is a text-only match from database
                
            Returns:
                "Self Database Trademark" or "Indian Trademark Journal"
            """
            # If it's a text-only match, it's definitely from Self Database
            if is_text_match:
                return "Self Database Trademark"
            
            # Check if trademark_id is numeric (application_id)
            try:
                app_id = int(trademark_id)
                # First check if it's in our text_results_map (already matched)
                if app_id in database_app_ids:
                    return "Self Database Trademark"
                
                # If not in text_results_map, check if it exists in database
                # This handles cases where vector search found a trademark that exists in DB
                # but didn't match via text similarity
                mark = application_queries.get_mark_by_application_id(app_id)
                if mark is not None:
                    return "Self Database Trademark"
            except (ValueError, TypeError):
                # Not a numeric ID, likely from vector database (UUID or other format)
                pass
            
            # Default to Indian Trademark Journal (from vector database)
            return "Indian Trademark Journal"
        
        # Combine vector and text results
        combined_results = []
        result_ids_seen = set()
        
        # Process vector results
        for vec_result in vector_results:
            trademark_id = vec_result['trademark_id']
            vector_score = vec_result['similarity_score']
            
            # Get text similarity if available
            text_score = None
            matched_mark = None
            is_text_match = False
            
            # Try to match by application_id (if trademark_id is numeric)
            try:
                app_id = int(trademark_id)
                if app_id in text_results_map:
                    text_score = text_results_map[app_id]['score']
                    matched_mark = text_results_map[app_id]['mark']
                    is_text_match = True
            except (ValueError, TypeError):
                # If trademark_id is not numeric, try to find by mark in metadata
                if extracted_text and text_similarity:
                    metadata = vec_result.get('metadata', {})
                    mark_in_metadata = metadata.get('mark') or metadata.get('name', '')
                    is_text_only = metadata.get('trademark_type') == 'text_only' or metadata.get('extraction_method') == 'text_only'
                    
                    if mark_in_metadata:
                        text_score = text_similarity.calculate_similarity(
                            extracted_text,
                            mark_in_metadata,
                            weights={
                                'levenshtein': Config.LEVENSHTEIN_WEIGHT,
                                'jaro_winkler': Config.JARO_WINKLER_WEIGHT,
                                'token_sort': Config.TOKEN_SORT_WEIGHT,
                                'phonetic': Config.PHONETIC_WEIGHT
                            }
                        )
                        
                        # For text-only trademarks, use a lower threshold since image-to-text embedding similarity might be lower
                        # But text-to-text similarity should be high
                        threshold_to_use = Config.TEXT_SIMILARITY_THRESHOLD * 0.7 if is_text_only else Config.TEXT_SIMILARITY_THRESHOLD
                        
                        if text_score >= threshold_to_use:
                            matched_mark = mark_in_metadata
                            # For text-only trademarks with good text match, boost the combined score
                            if is_text_only and text_score >= 0.7:
                                # If text similarity is high, we can trust it even if vector score is lower
                                pass
            
            # Check if this is a text-only trademark
            metadata = vec_result.get('metadata', {})
            is_text_only = metadata.get('trademark_type') == 'text_only' or metadata.get('extraction_method') == 'text_only'
            
            # Calculate combined score
            if text_score is not None:
                # For text-only trademarks, weight text similarity more heavily
                if is_text_only:
                    # Text-only: weight text similarity more (60% text, 40% vector)
                    combined_score = (
                        vector_score * 0.4 +
                        text_score * 0.6
                    )
                else:
                    # Image-based: use standard weights
                    combined_score = (
                        vector_score * Config.VECTOR_SIMILARITY_WEIGHT +
                        text_score * Config.TEXT_SIMILARITY_WEIGHT
                    )
            else:
                combined_score = vector_score
            
            # Determine source
            source = determine_source(trademark_id, is_text_match)
            
            combined_results.append({
                'trademark_id': trademark_id,
                'similarity_score': combined_score,
                'vector_similarity_score': vector_score,
                'text_similarity_score': text_score,
                'matched_mark': matched_mark or metadata.get('mark', ''),
                'source': source,
                'trademark_class': metadata.get('trademark_class'),
                'applicant_name': metadata.get('applicant_name'),
                'application_no': metadata.get('application_no'),
                'trademark_type': 'text_only' if is_text_only else 'image_based',
                'has_image': not is_text_only,
                'metadata': metadata
            })
            
            result_ids_seen.add(trademark_id)
        
        # Add text-only results (if any high-scoring text matches not in vector results)
        # IMPORTANT: Even if vector search returned 0 results, we should still return text matches
        if text_results_map:
            for app_id, text_data in text_results_map.items():
                app_id_str = str(app_id)
                if app_id_str not in result_ids_seen:
                    # For text-only matches, use the text score directly (not weighted down)
                    # This ensures good text matches are returned even without vector matches
                    text_only_score = text_data['score']
                    
                    # Text-only matches are always from Self Database
                    source = "Self Database Trademark"
                    
                    combined_results.append({
                        'trademark_id': app_id_str,
                        'similarity_score': text_only_score,  # Use full text score for text-only matches
                        'vector_similarity_score': None,
                        'text_similarity_score': text_data['score'],
                        'matched_mark': text_data['mark'],
                        'source': source,
                        'trademark_class': None,  # Will be populated later for database trademarks
                        'applicant_name': None,  # Will be populated later for database trademarks
                        'application_no': None,  # Will be populated later for database trademarks
                        'trademark_type': 'text_only',
                        'has_image': False,
                        'metadata': {
                            'application_id': app_id,
                            'source': 'text_match_only'
                        }
                    })
        
        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Take top_k results
        final_results = combined_results[:top_k]
        
        # Filter by threshold - but use a lower threshold for text-only matches
        final_threshold = threshold or Config.SIMILARITY_THRESHOLD
        filtered_results = []
        for r in final_results:
            metadata = r.get('metadata', {})
            is_text_only = metadata.get('trademark_type') == 'text_only' or metadata.get('extraction_method') == 'text_only'
            
            # For text-only matches, use text similarity threshold instead
            if r.get('vector_similarity_score') is None and r.get('text_similarity_score') is not None:
                # Text-only match: use text similarity threshold
                if r['text_similarity_score'] >= Config.TEXT_SIMILARITY_THRESHOLD:
                    filtered_results.append(r)
            elif is_text_only:
                # Text-only trademark found via vector search: use lower threshold
                # Accept if either vector score or combined score meets threshold
                text_threshold = Config.TEXT_SIMILARITY_THRESHOLD * 0.7
                if (r.get('text_similarity_score', 0) >= text_threshold or 
                    r['similarity_score'] >= final_threshold * 0.7):
                    filtered_results.append(r)
            else:
                # Hybrid or vector-only match: use combined threshold
                if r['similarity_score'] >= final_threshold:
                    filtered_results.append(r)
        
        final_results = filtered_results
        
        logger.info(f"Hybrid search found {len(final_results)} final results")
        
        # Fetch trademark class information for Self Database Trademarks
        # Collect all application IDs from Self Database results
        database_app_ids_to_fetch = []
        for result in final_results:
            if result.get('source') == 'Self Database Trademark':
                try:
                    app_id = int(result['trademark_id'])
                    database_app_ids_to_fetch.append(app_id)
                except (ValueError, TypeError):
                    # Not a numeric ID, skip
                    pass
        
        # Batch fetch application details including trademark class
        applications_data = {}
        if database_app_ids_to_fetch:
            try:
                applications_data = application_queries.get_applications_by_ids(database_app_ids_to_fetch)
                logger.info(f"Fetched class, applicant, and application_no information for {len(applications_data)} applications")
            except Exception as e:
                logger.warning(f"Error fetching application details: {e}")
                applications_data = {}
        
        # Add trademark_class, applicant_name, and application_no to results
        for result in final_results:
            if result.get('source') == 'Self Database Trademark':
                try:
                    app_id = int(result['trademark_id'])
                    app_data = applications_data.get(app_id)
                    if app_data:
                        result['trademark_class'] = app_data.get('trademark_class') or ''
                        result['applicant_name'] = app_data.get('applicant_name') or ''
                        result['application_no'] = app_data.get('application_no') or ''
                    else:
                        result['trademark_class'] = ''
                        result['applicant_name'] = ''
                        result['application_no'] = ''
                except (ValueError, TypeError):
                    result['trademark_class'] = None
                    result['applicant_name'] = None
                    result['application_no'] = None
            else:
                # For Indian Trademark Journal, get class from metadata if available
                metadata = result.get('metadata', {})
                result['trademark_class'] = metadata.get('trademark_class') or None
                result['applicant_name'] = None  # Not available for Indian Trademark Journal
                result['application_no'] = None  # Not available for Indian Trademark Journal
        
        # Calculate query time
        query_time = (time.time() - start_time) * 1000  # in milliseconds
        
        logger.info(f"Search completed: {len(final_results)} results in {query_time:.2f}ms")
        
        return SearchResponse(
            query_time_ms=query_time,
            total_results=len(final_results),
            extracted_text=extracted_text,
            results=[SearchResult(**r) for r in final_results]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Text-based search endpoint
@app.post("/search-text", response_model=SearchResponse, tags=["Image Search"])
async def search_trademarks_by_text(
    request: TextSearchRequest
):
    """
    Search for similar trademarks using text input (hybrid vector + text similarity)
    
    Args:
        request: TextSearchRequest containing query_text, top_k, threshold, and optional trademark_class
        
    Returns:
        Search results with combined similarity scores
    """
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not available. Please ensure Qdrant is running.")
    
    if embedding_generator is None:
        raise HTTPException(status_code=503, detail="Embedding generator not available. Please restart the server.")
    
    import time
    start_time = time.time()
    
    try:
        query_text = request.query_text.strip()
        if not query_text or len(query_text) < 1:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        top_k = request.top_k
        threshold = request.threshold
        trademark_class = request.trademark_class
        
        logger.info(f"Text search query: '{query_text}'")
        
        # Generate text embedding for vector search
        text_embedding = embedding_generator.generate_text_embedding(query_text)
        logger.info(f"Generated text embedding for query: '{query_text}'")
        
        # Vector similarity search using text embedding
        filter_dict = {}
        if trademark_class:
            filter_dict['trademark_class'] = trademark_class
        
        search_threshold = threshold or Config.SIMILARITY_THRESHOLD
        logger.info(f"Vector search with threshold: {search_threshold}")
        
        vector_results = vector_db.search_similar(
            query_embedding=text_embedding,
            top_k=top_k * 2,  # Get more results for hybrid scoring
            score_threshold=search_threshold * 0.8,  # Slightly lower threshold for text-to-text/image matching
            filter_dict=filter_dict if filter_dict else None
        )
        
        logger.info(f"Vector search found {len(vector_results)} results")
        
        # Text similarity search against MySQL database marks
        text_results_map = {}
        mark_text_to_app_id = {}  # Map mark text to application_id for lookup
        
        if text_similarity:
            try:
                # Get all marks from database
                marks = application_queries.get_all_marks(use_cache=True)
                logger.info(f"Comparing query text '{query_text}' against {len(marks)} marks from database")
                
                # Create mark text to app_id mapping
                mark_texts = []
                for mark_data in marks:
                    if not mark_data.get('mark'):
                        continue
                    mark_text = mark_data['mark']
                    app_id = mark_data['application_id']
                    mark_texts.append(mark_text)
                    mark_text_to_app_id[mark_text] = app_id
                
                # Find best text matches using optimized method
                text_matches = text_similarity.find_best_matches(
                    query_text=query_text,
                    candidate_texts=mark_texts,
                    top_k=top_k * 2,
                    threshold=Config.TEXT_SIMILARITY_THRESHOLD
                )
                
                # Create map of application_id -> text similarity score
                for matched_text, text_score in text_matches:
                    app_id = mark_text_to_app_id.get(matched_text)
                    if app_id:
                        text_results_map[app_id] = {
                            'score': text_score,
                            'mark': matched_text
                        }
                
                logger.info(f"Text similarity found {len(text_results_map)} matches above threshold")
                
                # If no matches found with fuzzy matching, try exact/partial match as fallback
                if len(text_results_map) == 0:
                    logger.info("No fuzzy matches found, trying exact/partial match fallback...")
                    normalized_query = text_similarity.normalize_text(query_text)
                    
                    for mark_data in marks:
                        if not mark_data.get('mark'):
                            continue
                        mark_text = mark_data['mark']
                        app_id = mark_data['application_id']
                        
                        # Normalize mark text
                        normalized_mark = text_similarity.normalize_text(mark_text)
                        
                        # Check for exact match (case-insensitive)
                        if normalized_query == normalized_mark:
                            text_results_map[app_id] = {
                                'score': 1.0,
                                'mark': mark_text
                            }
                            logger.info(f"Found exact match: '{mark_text}' (app_id: {app_id})")
                        # Check if query text is contained in mark or vice versa
                        elif normalized_query in normalized_mark or normalized_mark in normalized_query:
                            # Calculate similarity for partial matches
                            text_score = text_similarity.calculate_similarity(
                                query_text,
                                mark_text,
                                weights={
                                    'levenshtein': Config.LEVENSHTEIN_WEIGHT,
                                    'jaro_winkler': Config.JARO_WINKLER_WEIGHT,
                                    'token_sort': Config.TOKEN_SORT_WEIGHT,
                                    'phonetic': Config.PHONETIC_WEIGHT
                                }
                            )
                            if text_score >= 0.5:  # Lower threshold for partial matches
                                text_results_map[app_id] = {
                                    'score': text_score,
                                    'mark': mark_text
                                }
                                logger.info(f"Found partial match: '{mark_text}' (app_id: {app_id}, score: {text_score:.2f})")
                    
                    if text_results_map:
                        logger.info(f"Fallback search found {len(text_results_map)} matches")
                        
            except Exception as e:
                logger.error(f"Text similarity search error: {e}", exc_info=True)
                text_results_map = {}
        
        # Helper function to determine source of trademark
        database_app_ids = set(text_results_map.keys()) if text_results_map else set()
        
        def determine_source(trademark_id: str, is_text_match: bool = False) -> str:
            """Determine the source of a trademark"""
            if is_text_match:
                return "Self Database Trademark"
            
            try:
                app_id = int(trademark_id)
                if app_id in database_app_ids:
                    return "Self Database Trademark"
                mark = application_queries.get_mark_by_application_id(app_id)
                if mark is not None:
                    return "Self Database Trademark"
            except (ValueError, TypeError):
                pass
            
            return "Indian Trademark Journal"
        
        # Combine vector and text results
        combined_results = []
        result_ids_seen = set()
        
        # Process vector results
        for vec_result in vector_results:
            trademark_id = vec_result['trademark_id']
            vector_score = vec_result['similarity_score']
            
            # Get text similarity if available
            text_score = None
            matched_mark = None
            is_text_match = False
            
            # Try to match by application_id (if trademark_id is numeric)
            try:
                app_id = int(trademark_id)
                if app_id in text_results_map:
                    text_score = text_results_map[app_id]['score']
                    matched_mark = text_results_map[app_id]['mark']
                    is_text_match = True
            except (ValueError, TypeError):
                # If trademark_id is not numeric, try to find by mark in metadata
                if text_similarity:
                    metadata = vec_result.get('metadata', {})
                    mark_in_metadata = metadata.get('mark') or metadata.get('name', '')
                    is_text_only = metadata.get('trademark_type') == 'text_only' or metadata.get('extraction_method') == 'text_only'
                    
                    if mark_in_metadata:
                        text_score = text_similarity.calculate_similarity(
                            query_text,
                            mark_in_metadata,
                            weights={
                                'levenshtein': Config.LEVENSHTEIN_WEIGHT,
                                'jaro_winkler': Config.JARO_WINKLER_WEIGHT,
                                'token_sort': Config.TOKEN_SORT_WEIGHT,
                                'phonetic': Config.PHONETIC_WEIGHT
                            }
                        )
                        
                        threshold_to_use = Config.TEXT_SIMILARITY_THRESHOLD * 0.7 if is_text_only else Config.TEXT_SIMILARITY_THRESHOLD
                        
                        if text_score >= threshold_to_use:
                            matched_mark = mark_in_metadata
            
            # Check if this is a text-only trademark
            metadata = vec_result.get('metadata', {})
            is_text_only = metadata.get('trademark_type') == 'text_only' or metadata.get('extraction_method') == 'text_only'
            
            # Calculate combined score
            if text_score is not None:
                # For text-only trademarks, weight text similarity more heavily
                if is_text_only:
                    # Text-only: weight text similarity more (60% text, 40% vector)
                    combined_score = (
                        vector_score * 0.4 +
                        text_score * 0.6
                    )
                else:
                    # Image-based: use standard weights
                    combined_score = (
                        vector_score * Config.VECTOR_SIMILARITY_WEIGHT +
                        text_score * Config.TEXT_SIMILARITY_WEIGHT
                    )
            else:
                combined_score = vector_score
            
            # Determine source
            source = determine_source(trademark_id, is_text_match)
            
            combined_results.append({
                'trademark_id': trademark_id,
                'similarity_score': combined_score,
                'vector_similarity_score': vector_score,
                'text_similarity_score': text_score,
                'matched_mark': matched_mark or metadata.get('mark', ''),
                'source': source,
                'trademark_class': metadata.get('trademark_class'),
                'applicant_name': metadata.get('applicant_name'),
                'application_no': metadata.get('application_no'),
                'trademark_type': 'text_only' if is_text_only else 'image_based',
                'has_image': not is_text_only,
                'metadata': metadata
            })
            
            result_ids_seen.add(trademark_id)
        
        # Add text-only results (if any high-scoring text matches not in vector results)
        if text_results_map:
            for app_id, text_data in text_results_map.items():
                app_id_str = str(app_id)
                if app_id_str not in result_ids_seen:
                    text_only_score = text_data['score']
                    source = "Self Database Trademark"
                    
                    combined_results.append({
                        'trademark_id': app_id_str,
                        'similarity_score': text_only_score,
                        'vector_similarity_score': None,
                        'text_similarity_score': text_data['score'],
                        'matched_mark': text_data['mark'],
                        'source': source,
                        'trademark_class': None,
                        'applicant_name': None,
                        'application_no': None,
                        'trademark_type': 'text_only',
                        'has_image': False,
                        'metadata': {
                            'application_id': app_id,
                            'source': 'text_match_only'
                        }
                    })
        
        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Take top_k results
        final_results = combined_results[:top_k]
        
        # Filter by threshold
        final_threshold = threshold or Config.SIMILARITY_THRESHOLD
        filtered_results = []
        for r in final_results:
            metadata = r.get('metadata', {})
            is_text_only = metadata.get('trademark_type') == 'text_only' or metadata.get('extraction_method') == 'text_only'
            
            # For text-only matches, use text similarity threshold instead
            if r.get('vector_similarity_score') is None and r.get('text_similarity_score') is not None:
                if r['text_similarity_score'] >= Config.TEXT_SIMILARITY_THRESHOLD:
                    filtered_results.append(r)
            elif is_text_only:
                text_threshold = Config.TEXT_SIMILARITY_THRESHOLD * 0.7
                if (r.get('text_similarity_score', 0) >= text_threshold or 
                    r['similarity_score'] >= final_threshold * 0.7):
                    filtered_results.append(r)
            else:
                if r['similarity_score'] >= final_threshold:
                    filtered_results.append(r)
        
        final_results = filtered_results
        
        logger.info(f"Text search found {len(final_results)} final results")
        
        # Fetch trademark class information for Self Database Trademarks
        database_app_ids_to_fetch = []
        for result in final_results:
            if result.get('source') == 'Self Database Trademark':
                try:
                    app_id = int(result['trademark_id'])
                    database_app_ids_to_fetch.append(app_id)
                except (ValueError, TypeError):
                    pass
        
        # Batch fetch application details
        applications_data = {}
        if database_app_ids_to_fetch:
            try:
                applications_data = application_queries.get_applications_by_ids(database_app_ids_to_fetch)
                logger.info(f"Fetched class, applicant, and application_no information for {len(applications_data)} applications")
            except Exception as e:
                logger.warning(f"Error fetching application details: {e}")
                applications_data = {}
        
        # Add trademark_class, applicant_name, and application_no to results
        for result in final_results:
            if result.get('source') == 'Self Database Trademark':
                try:
                    app_id = int(result['trademark_id'])
                    app_data = applications_data.get(app_id)
                    if app_data:
                        result['trademark_class'] = app_data.get('trademark_class') or ''
                        result['applicant_name'] = app_data.get('applicant_name') or ''
                        result['application_no'] = app_data.get('application_no') or ''
                    else:
                        result['trademark_class'] = ''
                        result['applicant_name'] = ''
                        result['application_no'] = ''
                except (ValueError, TypeError):
                    result['trademark_class'] = None
                    result['applicant_name'] = None
                    result['application_no'] = None
            else:
                metadata = result.get('metadata', {})
                result['trademark_class'] = metadata.get('trademark_class') or None
                result['applicant_name'] = metadata.get('applicant_name') or None
                result['application_no'] = metadata.get('application_no') or None
        
        # Calculate query time
        query_time = (time.time() - start_time) * 1000  # in milliseconds
        
        logger.info(f"Text search completed: {len(final_results)} results in {query_time:.2f}ms")
        
        return SearchResponse(
            query_time_ms=query_time,
            total_results=len(final_results),
            extracted_text=query_text,
            results=[SearchResult(**r) for r in final_results]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text search error: {e}")
        raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")

# Get trademark by ID
@app.get("/trademark/{trademark_id}", response_model=TrademarkInfo, tags=["Image Search"])
async def get_trademark(trademark_id: str):
    """Get trademark information by ID"""
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not available. Please ensure Qdrant is running.")
    
    try:
        result = vector_db.get_trademark(trademark_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Trademark not found")
        
        return TrademarkInfo(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving trademark: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Database statistics
@app.get("/stats", response_model=DatabaseStats, tags=["Database Management"])
async def get_database_stats():
    """Get database statistics"""
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not available. Please ensure Qdrant is running.")
    
    try:
        stats = vector_db.get_collection_info()
        return DatabaseStats(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Refresh marks cache (for text similarity)
@app.post("/refresh-marks-cache", tags=["Database Management"])
async def refresh_marks_cache():
    """
    Refresh the cached marks used for text similarity.
    Use this after adding/updating marks in the database to make them available for search without restarting.
    """
    try:
        application_queries.refresh_cache()
        marks = application_queries.get_all_marks(use_cache=True)
        count = len(marks)
        logger.info(f"Marks cache refreshed: {count} marks loaded")
        return {"status": "ok", "cached_marks": count}
    except Exception as e:
        logger.error(f"Error refreshing marks cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh marks cache: {str(e)}")

# Batch search endpoint
@app.post("/batch-search", tags=["Image Search"])
async def batch_search_trademarks(
    files: List[UploadFile] = File(...),
    top_k: int = 10,
    threshold: Optional[float] = None
):
    """Search multiple trademarks at once"""
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not available. Please ensure Qdrant is running.")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            preprocessed = preprocessor.preprocess_pil(image)
            if preprocessed is None:
                results.append({"filename": file.filename, "error": "Preprocessing failed"})
                continue
            
            embedding = embedding_generator.generate_embedding(preprocessed)
            
            search_results = vector_db.search_similar(
                query_embedding=embedding,
                top_k=top_k,
                score_threshold=threshold or Config.SIMILARITY_THRESHOLD
            )
            
            results.append({
                "filename": file.filename,
                "results": search_results
            })
            
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    return {"batch_results": results}

# Delete trademark (admin)
@app.delete("/trademark/{trademark_id}", tags=["Database Management"])
async def delete_trademark(trademark_id: str):
    """Delete trademark from database"""
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not available. Please ensure Qdrant is running.")
    
    try:
        success = vector_db.delete_trademark(trademark_id)
        if success:
            logger.info(f"Deleted trademark: {trademark_id}")
            return {"message": f"Trademark {trademark_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Trademark not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting trademark: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# PDF Processing Endpoints

@app.post("/extract-pdf", response_model=PDFExtractionResult, tags=["PDF Processing"])
async def extract_trademarks_from_pdf(
    file: UploadFile = File(...),
    save_images: bool = True
):
    """
    Extract trademark images from a PDF file
    
    Args:
        file: Uploaded PDF file
        save_images: Whether to save extracted images to disk
        
    Returns:
        Extraction results with metadata
    """
    if pdf_extractor is None:
        raise HTTPException(status_code=503, detail="PDF extractor not available. Please restart the server.")
    
    import time
    import uuid
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Validate file size
        contents = await file.read()
        if len(contents) > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {Config.MAX_FILE_SIZE_MB}MB"
            )
        
        # Save PDF temporarily
        temp_pdf_path = Config.UPLOAD_PATH / f"temp_{uuid.uuid4().hex}.pdf"
        temp_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(contents)
        
        try:
            # Extract images from PDF
            extracted_images = pdf_extractor.extract_images_from_pdf(temp_pdf_path)
            
            # Prepare response data
            images_data = []
            saved_paths = []
            
            if save_images and extracted_images:
                # Save extracted images
                base_name = Path(file.filename).stem
                saved_paths = pdf_extractor.save_extracted_images(
                    extracted_images, 
                    Config.EXTRACTED_IMAGES_PATH, 
                    base_name
                )
            
            # Create image metadata
            for idx, (image, metadata) in enumerate(extracted_images):
                image_id = f"{Path(file.filename).stem}_img_{idx:03d}"
                
                image_data = ExtractedImage(
                    image_id=image_id,
                    width=metadata.get('width', image.width),
                    height=metadata.get('height', image.height),
                    page=metadata.get('page', 1),
                    extraction_method=metadata.get('source', 'unknown'),
                    metadata=metadata
                )
                images_data.append(image_data)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # logger.info(f"Extracted {len(extracted_images)} images from {file.filename} in {processing_time:.2f}ms")
            
            return PDFExtractionResult(
                pdf_filename=file.filename,
                total_images_extracted=len(extracted_images),
                images=images_data,
                processing_time_ms=processing_time
            )
            
        finally:
            # Clean up temporary PDF
            if temp_pdf_path.exists():
                temp_pdf_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

@app.post("/extract-pdf-batch", response_model=PDFBatchResult, tags=["PDF Processing"])
async def extract_trademarks_from_pdf_batch(
    files: List[UploadFile] = File(...),
    save_images: bool = True
):
    """
    Extract trademark images from multiple PDF files
    
    Args:
        files: List of uploaded PDF files (max 10)
        save_images: Whether to save extracted images to disk
        
    Returns:
        Batch extraction results
    """
    if pdf_extractor is None:
        raise HTTPException(status_code=503, detail="PDF extractor not available. Please restart the server.")
    
    import time
    import uuid
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 PDF files per batch")
    
    start_time = time.time()
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    
    results = []
    total_images = 0
    successful_pdfs = 0
    failed_pdfs = 0
    
    for file in files:
        try:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                failed_pdfs += 1
                results.append(PDFExtractionResult(
                    pdf_filename=file.filename,
                    total_images_extracted=0,
                    images=[],
                    processing_time_ms=0
                ))
                continue
            
            # Process single PDF (reuse the single PDF logic)
            contents = await file.read()
            
            # Save PDF temporarily
            temp_pdf_path = Config.UPLOAD_PATH / f"temp_{uuid.uuid4().hex}.pdf"
            temp_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_pdf_path, 'wb') as f:
                f.write(contents)
            
            try:
                # Extract images
                extracted_images = pdf_extractor.extract_images_from_pdf(temp_pdf_path)
                
                # Save images if requested
                if save_images and extracted_images:
                    base_name = Path(file.filename).stem
                    pdf_extractor.save_extracted_images(
                        extracted_images, 
                        Config.EXTRACTED_IMAGES_PATH, 
                        base_name
                    )
                
                # Create response data
                images_data = []
                for idx, (image, metadata) in enumerate(extracted_images):
                    image_id = f"{Path(file.filename).stem}_img_{idx:03d}"
                    
                    image_data = ExtractedImage(
                        image_id=image_id,
                        width=metadata.get('width', image.width),
                        height=metadata.get('height', image.height),
                        page=metadata.get('page', 1),
                        extraction_method=metadata.get('source', 'unknown'),
                        metadata=metadata
                    )
                    images_data.append(image_data)
                
                total_images += len(extracted_images)
                successful_pdfs += 1
                
                results.append(PDFExtractionResult(
                    pdf_filename=file.filename,
                    total_images_extracted=len(extracted_images),
                    images=images_data,
                    processing_time_ms=0  # Individual timing not tracked in batch
                ))
                
            finally:
                # Clean up temporary PDF
                if temp_pdf_path.exists():
                    temp_pdf_path.unlink()
                    
        except Exception as e:
            failed_pdfs += 1
            logger.error(f"Error processing PDF {file.filename}: {e}")
            results.append(PDFExtractionResult(
                pdf_filename=file.filename,
                total_images_extracted=0,
                images=[],
                processing_time_ms=0
            ))
    
    # Calculate total processing time
    processing_time = (time.time() - start_time) * 1000
    
    logger.info(f"Batch {batch_id}: {successful_pdfs} successful, {failed_pdfs} failed, {total_images} total images")
    
    return PDFBatchResult(
        batch_id=batch_id,
        total_pdfs=len(files),
        successful_pdfs=successful_pdfs,
        failed_pdfs=failed_pdfs,
        total_images_extracted=total_images,
        processing_time_ms=processing_time,
        results=results
    )

@app.get("/extracted-images", tags=["PDF Processing"])
async def list_extracted_images():
    """List all extracted trademark images"""
    try:
        extracted_dir = Config.EXTRACTED_IMAGES_PATH
        if not extracted_dir.exists():
            return {"images": [], "total": 0}
        
        image_files = []
        for file_path in extracted_dir.glob("*.png"):
            stat = file_path.stat()
            image_files.append({
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "created_at": stat.st_ctime,
                "path": str(file_path)
            })
        
        # Sort by creation time (newest first)
        image_files.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "images": image_files,
            "total": len(image_files),
            "directory": str(extracted_dir)
        }
        
    except Exception as e:
        logger.error(f"Error listing extracted images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index-extracted-images", tags=["PDF Processing"])
async def index_extracted_images(
    image_ids: Optional[List[str]] = None,
    auto_generate_metadata: bool = True
):
    """
    Index extracted images into the vector database
    
    Args:
        image_ids: List of specific image IDs to index (if None, index all)
        auto_generate_metadata: Whether to auto-generate metadata for images
        
    Returns:
        Indexing results
    """
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not available. Please ensure Qdrant is running.")
    
    import time
    start_time = time.time()
    
    try:
        extracted_dir = Config.EXTRACTED_IMAGES_PATH
        if not extracted_dir.exists():
            return {"message": "No extracted images directory found", "indexed": 0}
        
        # Get list of images to process
        if image_ids:
            image_files = []
            for image_id in image_ids:
                # Find matching files
                for pattern in [f"*{image_id}*", f"{image_id}*"]:
                    matches = list(extracted_dir.glob(pattern))
                    image_files.extend(matches)
        else:
            image_files = list(extracted_dir.glob("*.png"))
        
        if not image_files:
            return {"message": "No images found to index", "indexed": 0}
        
        indexed_count = 0
        failed_count = 0
        errors = []
        
        for image_path in image_files:
            try:
                # Generate trademark ID from filename
                trademark_id = image_path.stem
                
                # Preprocess image
                preprocessed = preprocessor.preprocess(image_path)
                if preprocessed is None:
                    raise ValueError("Preprocessing failed")
                
                # Generate embedding
                embedding = embedding_generator.generate_embedding(preprocessed)
                
                # Prepare metadata
                metadata = {
                    'trademark_id': trademark_id,
                    'name': trademark_id,
                    'trademark_class': '',
                    'applicant_name': '',
                    'application_no': '',
                    'registration_date': '',
                    'owner': '',
                    'image_path': str(image_path),
                    'indexed_at': datetime.now().isoformat(),
                    'source': 'pdf_extraction'
                }
                
                # Insert into database
                success = vector_db.insert_trademark(
                    trademark_id=trademark_id,
                    embedding=embedding,
                    metadata=metadata
                )
                
                if success:
                    indexed_count += 1
                else:
                    failed_count += 1
                    errors.append(f"Failed to insert {trademark_id}")
                    
            except Exception as e:
                failed_count += 1
                errors.append(f"Error processing {image_path.name}: {str(e)}")
                logger.error(f"Error indexing {image_path}: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Indexed {indexed_count} images, {failed_count} failed in {processing_time:.2f}ms")
        
        return {
            "message": f"Indexing completed: {indexed_count} successful, {failed_count} failed",
            "indexed": indexed_count,
            "failed": failed_count,
            "total_processed": len(image_files),
            "processing_time_ms": processing_time,
            "errors": errors[:10]  # Limit error details
        }
        
    except Exception as e:
        logger.error(f"Error indexing extracted images: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/process-pdf-and-index", tags=["PDF Processing"])
async def process_pdf_and_index(
    file: UploadFile = File(...),
    save_images: bool = True,
    auto_index: bool = True
):
    """
    Extract trademarks from PDF and optionally index them into the vector database
    
    Args:
        file: Uploaded PDF file
        save_images: Whether to save extracted images to disk
        auto_index: Whether to automatically index extracted images
        
    Returns:
        Combined extraction and indexing results
    """
    if pdf_extractor is None:
        raise HTTPException(status_code=503, detail="PDF extractor not available. Please restart the server.")
    
    if vector_db is None and auto_index:
        raise HTTPException(status_code=503, detail="Vector database not available. Please ensure Qdrant is running.")
    
    import time
    start_time = time.time()
    
    try:
        # Extract images directly (bypass the separate endpoint)
        import uuid
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Validate file size
        contents = await file.read()
        if len(contents) > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {Config.MAX_FILE_SIZE_MB}MB"
            )
        
        # Save PDF temporarily
        temp_pdf_path = Config.UPLOAD_PATH / f"temp_{uuid.uuid4().hex}.pdf"
        temp_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(contents)
        
        try:
            # Extract images from PDF
            extracted_images = pdf_extractor.extract_images_from_pdf(temp_pdf_path)
            
            # Extract text-only trademarks (trademarks without images)
            text_only_trademarks = pdf_extractor.extract_text_only_trademarks(temp_pdf_path)
            
            # Filter out text-only trademarks that already have corresponding images
            # (to avoid duplicates - if a page has both image and text, we prefer the image)
            pages_with_images = {metadata.get('page', 1) for _, metadata in extracted_images}
            filtered_text_trademarks = [
                tm for tm in text_only_trademarks 
                if tm.get('page', 1) not in pages_with_images or not tm.get('application_no')
            ]
            
            logger.info(f"Extracted {len(extracted_images)} images and {len(filtered_text_trademarks)} text-only trademarks")
            
            # Prepare response data
            images_data = []
            saved_paths = []
            
            if save_images and extracted_images:
                # Save extracted images
                # Use api/data/extracted directory (where files should be stored)
                api_dir = Path(__file__).parent / "api"  # api directory
                extracted_dir = (api_dir / "data" / "extracted").resolve()  # Resolve to absolute path
                extracted_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Saving images to: {extracted_dir}")
                logger.info(f"Extracted directory exists: {extracted_dir.exists()}")
                
                base_name = Path(file.filename).stem
                saved_paths = pdf_extractor.save_extracted_images(
                    extracted_images, 
                    extracted_dir, 
                    base_name
                )
                
                logger.info(f"Saved {len(saved_paths)} images to {extracted_dir}")
                if saved_paths:
                    logger.info(f"First saved image path: {saved_paths[0]}")
                    logger.info(f"First saved image exists: {saved_paths[0].exists() if saved_paths else False}")
            
            # Generate UUIDs for all images first
            image_uuids = [str(uuid.uuid4()) for _ in extracted_images]
            
            # Create image metadata
            for idx, (image, metadata) in enumerate(extracted_images):
                image_uuid = image_uuids[idx]
                image_id = f"{Path(file.filename).stem}_img_{idx:03d}"
                
                image_data = ExtractedImage(
                    image_id=image_uuid,  # Use UUID as the actual ID
                    width=metadata.get('width', image.width),
                    height=metadata.get('height', image.height),
                    page=metadata.get('page', 1),
                    extraction_method=metadata.get('source', 'unknown'),
                    metadata={
                        **metadata,
                        'original_filename': image_id,  # Keep original name in metadata
                        'pdf_source': file.filename
                    }
                )
                images_data.append(image_data)
            
            # Calculate extraction time
            extraction_time = (time.time() - start_time) * 1000
            
            # Now handle indexing
            indexed_count = 0
            indexing_errors = []
            text_indexed_count = 0
            text_indexing_errors = []
            
            if auto_index and extracted_images:
                logger.info(f"Starting indexing of {len(extracted_images)} images...")
                
                for idx, (image, metadata) in enumerate(extracted_images):
                    try:
                        # Use the same UUID that was generated earlier
                        image_uuid = image_uuids[idx]
                        image_id = f"{Path(file.filename).stem}_img_{idx:03d}"
                        
                        if save_images and saved_paths:
                            # Use the saved image file
                            image_path = saved_paths[idx]
                        else:
                            # Process the image directly from memory
                            image_path = None
                        
                        # Preprocess and generate embedding
                        if image_path and image_path.exists():
                            preprocessed = preprocessor.preprocess(image_path)
                        else:
                            # Process image directly from PIL Image
                            preprocessed = preprocessor.preprocess_pil(image)
                        
                        if preprocessed is None:
                            indexing_errors.append(f"Preprocessing failed for {image_id}")
                            continue
                        
                        embedding = embedding_generator.generate_embedding(preprocessed)
                        
                        # Prepare metadata
                        metadata_dict = {
                            'trademark_id': image_uuid,
                            'name': image_id,  # Keep human-readable name
                            'trademark_class': metadata.get('trademark_class', ''),
                            'applicant_name': metadata.get('applicant_name', ''),
                            'application_no': metadata.get('application_no', ''),
                            'registration_date': '',
                            'owner': metadata.get('applicant_name', ''),  # Use applicant_name as owner if available
                            'image_path': str(image_path) if image_path else 'in_memory',
                            'indexed_at': datetime.now().isoformat(),
                            'source': 'pdf_extraction',
                            'pdf_source': file.filename,
                            'page': metadata.get('page', 1),
                            'extraction_method': metadata.get('source', 'unknown'),
                            'width': metadata.get('width', image.width),
                            'height': metadata.get('height', image.height),
                            'original_filename': image_id
                        }
                        
                        # Insert into database
                        success = vector_db.insert_trademark(
                            trademark_id=image_uuid,  # Use UUID as the point ID
                            embedding=embedding,
                            metadata=metadata_dict
                        )
                        
                        if success:
                            indexed_count += 1
                            logger.info(f"Successfully indexed {image_id} (UUID: {image_uuid})")
                        else:
                            indexing_errors.append(f"Database insertion failed for {image_id}")
                            
                    except Exception as e:
                        indexing_errors.append(f"Error indexing {image_id}: {str(e)}")
                        logger.error(f"Error indexing {image_id}: {e}")
                        
                logger.info(f"Indexing completed: {indexed_count} successful, {len(indexing_errors)} failed")
            
            # Now handle text-only trademarks indexing
            text_indexed_count = 0
            text_indexing_errors = []
            
            if auto_index and filtered_text_trademarks:
                logger.info(f"Starting indexing of {len(filtered_text_trademarks)} text-only trademarks...")
                
                for tm_metadata in filtered_text_trademarks:
                    try:
                        # Generate UUID for this text-only trademark
                        text_uuid = str(uuid.uuid4())
                        
                        # Use application_no as identifier if available, otherwise use mark name
                        if tm_metadata.get('application_no'):
                            tm_id = f"{Path(file.filename).stem}_text_{tm_metadata['application_no']}"
                        elif tm_metadata.get('mark'):
                            # Create ID from mark name (sanitized)
                            mark_id = re.sub(r'[^a-zA-Z0-9]', '_', tm_metadata['mark'])[:50]
                            tm_id = f"{Path(file.filename).stem}_text_{mark_id}"
                        else:
                            tm_id = f"{Path(file.filename).stem}_text_{text_uuid[:8]}"
                        
                        # Get mark name for embedding
                        mark_name = tm_metadata.get('mark', '')
                        if not mark_name:
                            # If no mark name extracted, skip this trademark
                            text_indexing_errors.append(f"No mark name found for application {tm_metadata.get('application_no', 'unknown')}")
                            continue
                        
                        # Generate text embedding using CLIP
                        embedding = embedding_generator.generate_text_embedding(mark_name)
                        
                        # Prepare metadata
                        metadata_dict = {
                            'trademark_id': text_uuid,
                            'name': mark_name,  # Use mark name as the name
                            'mark': mark_name,
                            'trademark_class': tm_metadata.get('trademark_class', ''),
                            'applicant_name': tm_metadata.get('applicant_name', ''),
                            'application_no': tm_metadata.get('application_no', ''),
                            'registration_date': '',
                            'owner': tm_metadata.get('applicant_name', ''),
                            'image_path': '',  # No image for text-only trademarks
                            'indexed_at': datetime.now().isoformat(),
                            'source': 'pdf_text_extraction',
                            'pdf_source': file.filename,
                            'page': tm_metadata.get('page', 1),
                            'extraction_method': 'text_only',
                            'trademark_type': 'text_only'
                        }
                        
                        # Insert into database
                        success = vector_db.insert_trademark(
                            trademark_id=text_uuid,
                            embedding=embedding,
                            metadata=metadata_dict
                        )
                        
                        if success:
                            text_indexed_count += 1
                            logger.info(f"Successfully indexed text-only trademark: {mark_name} (UUID: {text_uuid})")
                        else:
                            text_indexing_errors.append(f"Database insertion failed for {tm_id}")
                            
                    except Exception as e:
                        text_indexing_errors.append(f"Error indexing text trademark {tm_metadata.get('mark', 'unknown')}: {str(e)}")
                        logger.error(f"Error indexing text-only trademark: {e}")
                
                logger.info(f"Text-only indexing completed: {text_indexed_count} successful, {len(text_indexing_errors)} failed")
        
        finally:
            # Clean up temporary PDF
            if temp_pdf_path.exists():
                temp_pdf_path.unlink()
        
        total_processing_time = (time.time() - start_time) * 1000
        
        return {
            "pdf_filename": file.filename,
            "total_images_extracted": len(extracted_images),
            "images_indexed": indexed_count,
            "text_only_trademarks_found": len(filtered_text_trademarks),
            "text_only_trademarks_indexed": text_indexed_count,
            "indexing_errors": len(indexing_errors),
            "text_indexing_errors": len(text_indexing_errors),
            "total_processing_time_ms": total_processing_time,
            "extraction_time_ms": extraction_time,
            "indexing_time_ms": total_processing_time - extraction_time,
            "images": images_data,
            "indexing_error_details": indexing_errors[:10],  # Limit error details
            "text_indexing_error_details": text_indexing_errors[:10]  # Limit error details
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process-pdf-and-index: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

        # Save PDF temporarily
        # Validate file type
# Serve trademark images
@app.get("/image/{trademark_id}", tags=["Image Search"])
async def get_trademark_image(trademark_id: str):
    """
    Serve trademark image by ID
    
    Args:
        trademark_id: Trademark ID to get image for
        
    Returns:
        Image file or 404 if not found
    """
    try:
        # Try multiple possible extracted directory locations
        # Files are actually stored in api/data/extracted based on find results
        project_root = Path(__file__).parent  # project root
        api_dir = project_root / "api"  # api directory
        
        # Possible extracted directory locations
        possible_dirs = [
            api_dir / "data" / "extracted",  # api/data/extracted (where files actually are)
            project_root / "data" / "extracted",  # data/extracted (config default)
            Config.EXTRACTED_IMAGES_PATH.resolve(),  # From config
        ]
        
        extracted_dir = None
        for possible_dir in possible_dirs:
            if possible_dir.exists() and any(possible_dir.glob("*.png")):
                extracted_dir = possible_dir
                break
        
        if not extracted_dir:
            extracted_dir = api_dir / "data" / "extracted"  # Default to api/data/extracted
        
        # First, try to get metadata from database to find image by name
        if vector_db:
            trademark_data = vector_db.get_trademark(trademark_id)
            if trademark_data and trademark_data.get('metadata'):
                metadata = trademark_data['metadata']
                
                # First try to use the image_path if available (absolute or relative path)
                if 'image_path' in metadata:
                    image_path_str = metadata['image_path']
                    
                    # Skip if it's 'in_memory'
                    if image_path_str and image_path_str != 'in_memory':
                        # Extract just the filename from the path
                        # Handle paths like "data/extracted/filename.png" or just "filename.png"
                        if '/' in image_path_str or '\\' in image_path_str:
                            # Path contains directory separators, extract filename
                            filename = Path(image_path_str).name
                        else:
                            # Already just a filename
                            filename = image_path_str
                        
                        # Try to find the file in extracted_dir
                        image_path = extracted_dir / filename
                        if image_path.exists():
                            return FileResponse(
                                path=str(image_path),
                                media_type="image/png" if image_path.suffix.lower() == '.png' else "image/jpeg",
                                filename=image_path.name
                            )
                        
                        # Also try resolving the path relative to api directory
                        # Handle "data/extracted/filename.png" relative to api directory
                        if "data/extracted/" in image_path_str:
                            api_data_path = api_dir / "data" / "extracted" / filename
                            if api_data_path.exists():
                                return FileResponse(
                                    path=str(api_data_path),
                                    media_type="image/png" if api_data_path.suffix.lower() == '.png' else "image/jpeg",
                                    filename=api_data_path.name
                                )
                        
                        # Try relative to project root
                        if image_path_str.startswith("data/extracted/"):
                            project_data_path = project_root / "data" / "extracted" / filename
                            if project_data_path.exists():
                                return FileResponse(
                                    path=str(project_data_path),
                                    media_type="image/png" if project_data_path.suffix.lower() == '.png' else "image/jpeg",
                                    filename=project_data_path.name
                                )
                
                # Try image_filename field
                if 'image_filename' in metadata:
                    image_filename = metadata['image_filename']
                    image_path = extracted_dir / image_filename
                    if image_path.exists():
                        return FileResponse(
                            path=str(image_path),
                            media_type="image/png" if image_path.suffix.lower() == '.png' else "image/jpeg",
                            filename=image_path.name
                        )
                
                # Try to find by original_filename or name (which may contain the base filename pattern)
                search_terms = []
                if 'original_filename' in metadata:
                    search_terms.append(metadata['original_filename'])
                if 'name' in metadata:
                    search_terms.append(metadata['name'])
                
                for search_term in search_terms:
                    if not search_term:
                        continue
                    
                    # Extract base name from search_term (e.g., "Screenshot 2025-07-22 at 7.19.25 PM_img_000" -> "Screenshot 2025-07-22 at 7.19.25 PM")
                    # Also handle format like "filename_img_000" -> "filename"
                    base_name = search_term
                    if '_img_' in search_term:
                        # Split at _img_ to get base name
                        base_name = search_term.split('_img_')[0]
                    elif 'img_' in search_term:
                        base_name = search_term.split('img_')[0].rstrip('_')
                    
                    # Try multiple search patterns
                    search_patterns = [
                        f"{base_name}_page*_img*.png",  # Match pattern: base_page*_img*.png
                        f"{base_name}_page*_img*.jpg",
                        f"{base_name}_page*_img*.jpeg",
                        f"{base_name}*_img*.png",  # More flexible pattern
                        f"{base_name}*_img*.jpg",
                        f"{base_name}*_img*.jpeg",
                        f"*{base_name}*_img*.png",  # Even more flexible
                        f"*{base_name}*_img*.jpg",
                        f"*{base_name}*_img*.jpeg",
                        f"*{base_name}*.png",  # Fallback: any file with base name
                        f"*{base_name}*.jpg",
                        f"*{base_name}*.jpeg",
                    ]
                    
                    for pattern in search_patterns:
                        matching_files = list(extracted_dir.glob(pattern))
                        if matching_files:
                            # Return the first match (or could return closest match)
                            file_path = matching_files[0]
                            return FileResponse(
                                path=str(file_path),
                                media_type="image/png" if file_path.suffix.lower() == '.png' else "image/jpeg",
                                filename=file_path.name
                            )
        
        # Fallback: try to find by trademark_id in filename (in case id matches filename)
        if extracted_dir.exists():
            for file_path in extracted_dir.glob("*.png"):
                if trademark_id in file_path.stem:
                    return FileResponse(
                        path=str(file_path),
                        media_type="image/png",
                        filename=file_path.name
                    )
            for file_path in extracted_dir.glob("*.jpg"):
                if trademark_id in file_path.stem:
                    return FileResponse(
                        path=str(file_path),
                        media_type="image/jpeg",
                        filename=file_path.name
                    )
            for file_path in extracted_dir.glob("*.jpeg"):
                if trademark_id in file_path.stem:
                    return FileResponse(
                        path=str(file_path),
                        media_type="image/jpeg",
                        filename=file_path.name
                    )
        raise HTTPException(status_code=404, detail=f"Image not found for trademark_id: {trademark_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image {trademark_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

# Debug endpoint for similarity search testing
@app.post("/debug-similarity", tags=["Debug & Testing"])
async def debug_similarity_search(
    file: UploadFile = File(...),
    threshold: float = 0.0
):
    """
    Debug endpoint to test similarity search with detailed information
    """
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not available.")
    
    try:
        # Load and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        preprocessed = preprocessor.preprocess_pil(image)
        if preprocessed is None:
            raise HTTPException(status_code=400, detail="Failed to preprocess image")
        
        # Generate embedding
        embedding = embedding_generator.generate_embedding(preprocessed)
        
        # Get all points from database
        all_points = vector_db.client.scroll(
            collection_name=vector_db.collection_name,
            limit=1000
        )[0]
        
        # Search with very low threshold to get all results
        results = vector_db.search_similar(
            query_embedding=embedding,
            top_k=50,
            score_threshold=threshold
        )
        
        # Calculate similarity with all stored points manually
        import numpy as np
        manual_similarities = []
        for point in all_points:
            if point.payload and 'trademark_id' in point.payload:
                # Calculate cosine similarity manually
                stored_embedding = np.array(point.vector)
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                # Clamp similarity to valid range [0, 1]
                similarity = max(0.0, min(1.0, float(similarity)))
                manual_similarities.append({
                    'trademark_id': point.payload.get('trademark_id'),
                    'name': point.payload.get('name', 'Unknown'),
                    'similarity': similarity,
                    'point_id': point.id
                })
        
        # Sort by similarity
        manual_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "query_embedding_shape": embedding.shape,
            "query_embedding_norm": float(np.linalg.norm(embedding)),
            "total_points_in_db": len(all_points),
            "search_results_count": len(results),
            "search_results": results[:10],
            "manual_similarities": manual_similarities[:10],
            "threshold_used": threshold,
            "config_threshold": Config.SIMILARITY_THRESHOLD
        }
        
    except Exception as e:
        logger.error(f"Debug similarity error: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")