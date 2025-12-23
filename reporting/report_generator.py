"""
Report Generator for Weekly Trademark Similarity Reports

Compares trademarks from Indian Trademark Journal (uploaded on Mondays)
against the self database to identify similarities.

Uses CLIP cross-modal similarity:
- Journal IMAGE trademarks are compared using their CLIP image embeddings
- Self DB TEXT marks are compared using their CLIP text embeddings
- Both exist in the same vector space, enabling image-to-text similarity
"""

import logging
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from config import Config
from database.vector_db import VectorDatabase
from database.application_queries import application_queries
from utils.text_similarity import TextSimilarity
from reporting.report_models import (
    SimilarTrademark,
    JournalTrademarkEntry,
    ReportSummary,
    WeeklyReport
)

logger = logging.getLogger(__name__)

# Source identifier for self database marks in Qdrant
SELF_DB_SOURCE = "self_database"


def extract_class_numbers(class_str: str) -> Set[str]:
    """
    Extract class numbers from a trademark class string.
    
    Handles various formats:
    - "5" -> {"5"}
    - "Class 5" -> {"5"}
    - "Class 5, Class 29" -> {"5", "29"}
    - "5, 29" -> {"5", "29"}
    - "Class 5, 29" -> {"5", "29"}
    
    Args:
        class_str: Class string in any format
        
    Returns:
        Set of class numbers as strings
    """
    if not class_str:
        return set()
    
    # Find all numbers in the string
    numbers = re.findall(r'\d+', class_str)
    return set(numbers)


def classes_match(class1: str, class2: str) -> bool:
    """
    Check if two trademark class strings have any matching classes.
    
    Args:
        class1: First class string (e.g., "5" or "Class 5")
        class2: Second class string (e.g., "Class 5" or "5, 29")
        
    Returns:
        True if any class numbers match, False otherwise
    """
    if not class1 or not class2:
        return False
    
    nums1 = extract_class_numbers(class1)
    nums2 = extract_class_numbers(class2)
    
    # Check for intersection
    return bool(nums1 & nums2)


class ReportGenerator:
    """Generates weekly similarity reports comparing journal trademarks against self database"""
    
    # Similarity thresholds for report generation
    # Text-to-text comparison uses higher threshold (same modality)
    TEXT_SIMILARITY_THRESHOLD = 0.5
    
    # Image-to-text (cross-modal) uses lower threshold because CLIP cross-modal
    # scores are naturally lower than same-modality comparisons
    VECTOR_SIMILARITY_THRESHOLD = 0.25
    
    # For reporting purposes, use the text threshold as the "official" threshold
    REPORT_SIMILARITY_THRESHOLD = 0.5
    
    def __init__(self, vector_db: VectorDatabase, text_similarity: Optional[TextSimilarity] = None):
        """
        Initialize the report generator
        
        Args:
            vector_db: VectorDatabase instance for querying journal trademarks
            text_similarity: TextSimilarity instance for text comparisons
        """
        self.vector_db = vector_db
        self.text_similarity = text_similarity or TextSimilarity(
            use_levenshtein=True,
            use_phonetic=True,
            use_fuzzywuzzy=True
        )
        # Use different thresholds for text vs vector similarity
        self.text_similarity_threshold = self.TEXT_SIMILARITY_THRESHOLD
        self.vector_similarity_threshold = self.VECTOR_SIMILARITY_THRESHOLD
        # For backward compatibility
        self.similarity_threshold = self.REPORT_SIMILARITY_THRESHOLD
    
    def generate_weekly_report(self, monday_date: Optional[datetime] = None) -> WeeklyReport:
        """
        Generate a weekly similarity report for journal trademarks uploaded on a specific Monday
        
        Args:
            monday_date: The Monday date to generate report for. If None, uses most recent Monday.
            
        Returns:
            WeeklyReport object containing all similarity data
        """
        start_time = time.time()
        
        # Calculate Monday date if not provided
        if monday_date is None:
            today = datetime.now()
            days_since_monday = today.weekday()
            monday_date = today - timedelta(days=days_since_monday)
        
        monday_date = monday_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        logger.info(f"Generating report for Monday: {monday_date.strftime('%Y-%m-%d')}")
        
        # Generate unique report ID
        report_id = f"report_{monday_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        
        # Step 1: Get all journal trademarks from Monday
        journal_trademarks = self._get_monday_journal_trademarks(monday_date)
        logger.info(f"Found {len(journal_trademarks)} journal trademarks from Monday")
        
        if not journal_trademarks:
            logger.warning("No journal trademarks found for the specified Monday")
            # Return empty report
            return WeeklyReport(
                report_id=report_id,
                monday_date=monday_date,
                summary=ReportSummary(
                    similarity_threshold_used=self.similarity_threshold
                ),
                processing_time_seconds=time.time() - start_time
            )
        
        # Step 2: Get all self database trademarks
        self_db_trademarks = self._get_self_database_trademarks()
        logger.info(f"Found {len(self_db_trademarks)} self database trademarks")
        
        # Step 3: Compare each journal trademark against self database
        entries = []
        total_similarities = 0
        all_similarity_scores = []
        
        for idx, journal_tm in enumerate(journal_trademarks):
            logger.info(f"Processing journal trademark {idx + 1}/{len(journal_trademarks)}")
            
            # Find similar trademarks
            similar_trademarks = self._find_similar_trademarks(journal_tm, self_db_trademarks)
            
            # Create journal entry
            entry = self._create_journal_entry(journal_tm, similar_trademarks)
            entries.append(entry)
            
            # Update statistics
            total_similarities += len(similar_trademarks)
            all_similarity_scores.extend([tm.similarity_score for tm in similar_trademarks])
        
        # Step 4: Calculate summary statistics
        summary = self._calculate_summary(
            entries=entries,
            self_db_count=len(self_db_trademarks),
            all_scores=all_similarity_scores
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Report generation completed in {processing_time:.2f} seconds")
        
        # Create and return report
        return WeeklyReport(
            report_id=report_id,
            monday_date=monday_date,
            summary=summary,
            entries=entries,
            processing_time_seconds=processing_time
        )
    
    def _get_monday_journal_trademarks(self, monday_date: datetime) -> List[Dict]:
        """
        Get all journal trademarks for the specified Monday date.
        Uses 'journal_monday_date' metadata field for exact matching.
        
        Args:
            monday_date: The Monday date to filter by
            
        Returns:
            List of trademark dictionaries from vector database
        """
        # First try to get by journal_monday_date field (preferred - exact match)
        trademarks = self.vector_db.get_trademarks_by_journal_monday(
            monday_date=monday_date,
            include_sources=['pdf_extraction', 'pdf_text_extraction']
        )
        
        # If no results, fall back to indexed_at date range (for backward compatibility)
        if not trademarks:
            logger.info("No trademarks found by journal_monday_date, falling back to indexed_at filter")
            trademarks = self.vector_db.get_trademarks_by_monday(
                monday_date=monday_date,
                include_sources=['pdf_extraction', 'pdf_text_extraction']
            )
        
        return trademarks
    
    def _get_self_database_trademarks(self) -> List[Dict]:
        """
        Get all trademarks from the self (MySQL) database
        
        Returns:
            List of trademark dictionaries with mark and application details
        """
        marks = application_queries.get_all_marks(use_cache=True)
        
        # Get full application details for all marks
        app_ids = [m['application_id'] for m in marks if m.get('application_id')]
        
        if not app_ids:
            return marks
        
        # Batch fetch application details
        app_details = application_queries.get_applications_by_ids(app_ids)
        
        # Merge mark data with application details
        enriched_marks = []
        for mark in marks:
            app_id = mark.get('application_id')
            if app_id and app_id in app_details:
                details = app_details[app_id]
                enriched_marks.append({
                    'application_id': app_id,
                    'mark': mark.get('mark', ''),
                    'trademark_class': details.get('trademark_class', ''),
                    'applicant_name': details.get('applicant_name', ''),
                    'application_no': details.get('application_no', '')
                })
            else:
                enriched_marks.append(mark)
        
        return enriched_marks
    
    def _find_similar_trademarks(self, journal_tm: Dict, 
                                  self_db_trademarks: List[Dict]) -> List[SimilarTrademark]:
        """
        Find similar trademarks in self database for a given journal trademark.
        
        Uses CLIP cross-modal similarity for image-based trademarks:
        - Journal image vector (CLIP image embedding) is compared against
        - Self DB text vectors (CLIP text embeddings) in Qdrant
        
        For text-only trademarks, uses traditional text similarity.
        
        Args:
            journal_tm: Journal trademark dictionary with metadata and vector
            self_db_trademarks: List of self database trademarks (used for fallback)
            
        Returns:
            List of SimilarTrademark objects above threshold
        """
        similar = []
        metadata = journal_tm.get('metadata', {})
        journal_vector = journal_tm.get('vector')
        
        # Get journal trademark text (mark name) for text comparison
        journal_mark = metadata.get('mark', '') or metadata.get('name', '')
        is_text_only = metadata.get('trademark_type') == 'text_only' or metadata.get('extraction_method') == 'text_only'
        
        # For IMAGE-BASED trademarks: Use vector similarity search (CLIP cross-modal)
        if not is_text_only and journal_vector is not None:
            similar = self._find_similar_by_vector(journal_tm, self_db_trademarks)
        else:
            # For TEXT-ONLY trademarks: Use text similarity
            similar = self._find_similar_by_text(journal_tm, self_db_trademarks)
        
        return similar
    
    def _find_similar_by_vector(self, journal_tm: Dict, 
                                 self_db_trademarks: List[Dict]) -> List[SimilarTrademark]:
        """
        Find similar trademarks using CLIP vector similarity search.
        
        This enables cross-modal comparison:
        - Journal IMAGE embedding vs Self DB TEXT embeddings
        
        Args:
            journal_tm: Journal trademark with vector
            self_db_trademarks: Fallback self DB trademarks for text comparison
            
        Returns:
            List of SimilarTrademark objects
        """
        similar = []
        metadata = journal_tm.get('metadata', {})
        journal_vector = journal_tm.get('vector')
        journal_mark = metadata.get('mark', '') or metadata.get('name', '')
        journal_class = metadata.get('trademark_class', '')
        
        if journal_vector is None:
            logger.warning("No vector found for journal trademark, falling back to text similarity")
            return self._find_similar_by_text(journal_tm, self_db_trademarks)
        
        # Convert to numpy array if needed
        if isinstance(journal_vector, list):
            journal_vector = np.array(journal_vector)
        
        try:
            # Search for similar vectors in self_database source
            # Use lower threshold for cross-modal (image→text) similarity
            vector_results = self.vector_db.search_similar_by_source(
                query_embedding=journal_vector,
                source=SELF_DB_SOURCE,
                top_k=20,  # Get more to allow filtering
                score_threshold=self.vector_similarity_threshold
            )
            
            if vector_results:
                logger.debug(f"Found {len(vector_results)} vector matches for journal trademark")
                
                for result in vector_results:
                    vector_score = result['similarity_score']
                    result_metadata = result.get('metadata', {})
                    self_mark = result_metadata.get('mark', '')
                    
                    # Calculate text similarity as secondary metric
                    text_score = None
                    if journal_mark and self_mark:
                        text_score = self.text_similarity.calculate_similarity(
                            journal_mark,
                            self_mark,
                            weights={
                                'levenshtein': Config.LEVENSHTEIN_WEIGHT,
                                'jaro_winkler': Config.JARO_WINKLER_WEIGHT,
                                'token_sort': Config.TOKEN_SORT_WEIGHT,
                                'phonetic': Config.PHONETIC_WEIGHT
                            }
                        )
                    
                    # Use vector score as primary (it's cross-modal CLIP similarity)
                    combined_score = vector_score
                    
                    # If we have both scores, use the higher one
                    # Text similarity might be higher if OCR extracted good text
                    if text_score is not None and text_score > vector_score:
                        combined_score = max(vector_score, text_score)
                    
                    # Use vector threshold for cross-modal comparison
                    if combined_score >= self.vector_similarity_threshold:
                        similar.append(SimilarTrademark(
                            self_db_application_id=result_metadata.get('application_id', 0),
                            mark=self_mark,
                            similarity_score=round(combined_score, 4),
                            vector_similarity_score=round(vector_score, 4),
                            text_similarity_score=round(text_score, 4) if text_score else None,
                            trademark_class=result_metadata.get('trademark_class', ''),
                            applicant_name=result_metadata.get('applicant_name', ''),
                            application_no=result_metadata.get('application_no', '')
                        ))
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            # Fall back to text similarity
            return self._find_similar_by_text(journal_tm, self_db_trademarks)
        
        # If no vector results, fall back to text similarity
        if not similar:
            logger.debug("No vector matches found, falling back to text similarity")
            similar = self._find_similar_by_text(journal_tm, self_db_trademarks)
        
        # Sort by: 1) Class match (same class first), 2) Similarity score (descending)
        # This prioritizes trademarks from the same class as the journal trademark
        # Use classes_match() to handle different class formats (e.g., "5" vs "Class 5")
        similar.sort(key=lambda x: (classes_match(x.trademark_class, journal_class), x.similarity_score), reverse=True)
        
        # Limit to top 10 similar trademarks
        return similar[:10]
    
    def _find_similar_by_text(self, journal_tm: Dict, 
                               self_db_trademarks: List[Dict]) -> List[SimilarTrademark]:
        """
        Find similar trademarks using text similarity.
        Used for text-only trademarks or as fallback.
        
        Args:
            journal_tm: Journal trademark dictionary
            self_db_trademarks: List of self database trademarks
            
        Returns:
            List of SimilarTrademark objects
        """
        similar = []
        metadata = journal_tm.get('metadata', {})
        journal_mark = metadata.get('mark', '') or metadata.get('name', '')
        journal_class = metadata.get('trademark_class', '')
        
        if not journal_mark:
            return similar
        
        for self_tm in self_db_trademarks:
            self_mark = self_tm.get('mark', '')
            if not self_mark:
                continue
            
            text_score = self.text_similarity.calculate_similarity(
                journal_mark,
                self_mark,
                weights={
                    'levenshtein': Config.LEVENSHTEIN_WEIGHT,
                    'jaro_winkler': Config.JARO_WINKLER_WEIGHT,
                    'token_sort': Config.TOKEN_SORT_WEIGHT,
                    'phonetic': Config.PHONETIC_WEIGHT
                }
            )
            
            if text_score is not None and text_score >= self.text_similarity_threshold:
                similar.append(SimilarTrademark(
                    self_db_application_id=self_tm.get('application_id', 0),
                    mark=self_mark,
                    similarity_score=round(text_score, 4),
                    vector_similarity_score=None,
                    text_similarity_score=round(text_score, 4),
                    trademark_class=self_tm.get('trademark_class', ''),
                    applicant_name=self_tm.get('applicant_name', ''),
                    application_no=self_tm.get('application_no', '')
                ))
        
        # Sort by: 1) Class match (same class first), 2) Similarity score (descending)
        # This prioritizes trademarks from the same class as the journal trademark
        # Use classes_match() to handle different class formats (e.g., "5" vs "Class 5")
        similar.sort(key=lambda x: (classes_match(x.trademark_class, journal_class), x.similarity_score), reverse=True)
        
        # Limit to top 10 similar trademarks
        return similar[:10]
    
    def _create_journal_entry(self, journal_tm: Dict, 
                               similar_trademarks: List[SimilarTrademark]) -> JournalTrademarkEntry:
        """
        Create a JournalTrademarkEntry from journal trademark data
        
        Args:
            journal_tm: Journal trademark dictionary
            similar_trademarks: List of similar trademarks found
            
        Returns:
            JournalTrademarkEntry object
        """
        metadata = journal_tm.get('metadata', {})
        
        # Determine trademark type
        trademark_type = 'text_only' if (
            metadata.get('trademark_type') == 'text_only' or 
            metadata.get('extraction_method') == 'text_only'
        ) else 'image_based'
        
        return JournalTrademarkEntry(
            journal_trademark_id=str(journal_tm.get('trademark_id', '')),
            trademark_type=trademark_type,
            mark_name=metadata.get('mark', '') or metadata.get('name', ''),
            trademark_class=metadata.get('trademark_class', ''),
            applicant_name=metadata.get('applicant_name', ''),
            application_no=metadata.get('application_no', ''),
            pdf_source=metadata.get('pdf_source', ''),
            page_number=metadata.get('page', 1),
            indexed_at=metadata.get('indexed_at', ''),
            journal_monday_date=metadata.get('journal_monday_date', ''),
            image_path=metadata.get('image_path') if trademark_type == 'image_based' else None,
            similar_trademarks=similar_trademarks
        )
    
    def _calculate_summary(self, entries: List[JournalTrademarkEntry],
                           self_db_count: int,
                           all_scores: List[float]) -> ReportSummary:
        """
        Calculate summary statistics for the report
        
        Args:
            entries: List of all journal trademark entries
            self_db_count: Number of self database trademarks
            all_scores: List of all similarity scores
            
        Returns:
            ReportSummary object
        """
        total_journal = len(entries)
        total_image_based = sum(1 for e in entries if e.trademark_type == 'image_based')
        total_text_only = sum(1 for e in entries if e.trademark_type == 'text_only')
        trademarks_with_similarities = sum(1 for e in entries if e.has_similarities)
        total_similarities = sum(len(e.similar_trademarks) for e in entries)
        
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        max_score = max(all_scores) if all_scores else 0.0
        
        return ReportSummary(
            total_journal_trademarks=total_journal,
            total_image_based=total_image_based,
            total_text_only=total_text_only,
            total_self_db_trademarks=self_db_count,
            total_similarities_found=total_similarities,
            trademarks_with_similarities=trademarks_with_similarities,
            average_similarity_score=round(avg_score, 4),
            highest_similarity_score=round(max_score, 4),
            similarity_threshold_used=self.REPORT_SIMILARITY_THRESHOLD  # Always 0.5
        )

