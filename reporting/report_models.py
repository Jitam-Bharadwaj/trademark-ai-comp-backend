"""
Pydantic models for report data structures
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SimilarTrademark(BaseModel):
    """A trademark from the self database that is similar to a journal trademark"""
    self_db_application_id: int = Field(..., description="Application ID from self database")
    mark: str = Field(..., description="Mark/name of the trademark")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Combined similarity score")
    vector_similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Vector-based similarity score")
    text_similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Text-based similarity score")
    trademark_class: str = Field(default="", description="Trademark class")
    applicant_name: str = Field(default="", description="Applicant name")
    application_no: str = Field(default="", description="Application number")


class JournalTrademarkEntry(BaseModel):
    """A trademark from the Indian Trademark Journal with its similar matches"""
    journal_trademark_id: str = Field(..., description="Trademark ID from vector database")
    trademark_type: str = Field(..., description="Type: 'image_based' or 'text_only'")
    mark_name: str = Field(default="", description="Mark name (for text-only or extracted)")
    trademark_class: str = Field(default="", description="Trademark class")
    applicant_name: str = Field(default="", description="Applicant name")
    application_no: str = Field(default="", description="Application number from journal")
    pdf_source: str = Field(default="", description="Source PDF filename")
    page_number: int = Field(default=1, description="Page number in PDF")
    indexed_at: str = Field(default="", description="When this trademark was indexed")
    journal_monday_date: str = Field(default="", description="Monday date this journal belongs to (YYYY-MM-DD)")
    image_path: Optional[str] = Field(None, description="Path to trademark image (if image-based)")
    similar_trademarks: List[SimilarTrademark] = Field(default_factory=list, description="List of similar trademarks from self DB")
    
    @property
    def has_similarities(self) -> bool:
        """Check if this entry has any similar trademarks"""
        return len(self.similar_trademarks) > 0
    
    @property
    def highest_similarity(self) -> float:
        """Get the highest similarity score among matches"""
        if not self.similar_trademarks:
            return 0.0
        return max(tm.similarity_score for tm in self.similar_trademarks)


class ReportSummary(BaseModel):
    """Summary statistics for the report"""
    total_journal_trademarks: int = Field(default=0, description="Total trademarks from journal")
    total_image_based: int = Field(default=0, description="Number of image-based trademarks")
    total_text_only: int = Field(default=0, description="Number of text-only trademarks")
    total_self_db_trademarks: int = Field(default=0, description="Total trademarks in self database compared")
    total_similarities_found: int = Field(default=0, description="Total similarity matches found")
    trademarks_with_similarities: int = Field(default=0, description="Number of journal trademarks that have at least one match")
    average_similarity_score: float = Field(default=0.0, description="Average similarity score across all matches")
    highest_similarity_score: float = Field(default=0.0, description="Highest similarity score found")
    similarity_threshold_used: float = Field(default=0.5, description="Similarity threshold used for matching")


class WeeklyReport(BaseModel):
    """Complete weekly report data structure"""
    report_id: str = Field(..., description="Unique report identifier")
    report_date: datetime = Field(default_factory=datetime.now, description="When the report was generated")
    monday_date: datetime = Field(..., description="The Monday date for journal uploads")
    journal_name: str = Field(default="", description="Journal name/number if available")
    summary: ReportSummary = Field(default_factory=ReportSummary, description="Report summary statistics")
    entries: List[JournalTrademarkEntry] = Field(default_factory=list, description="List of journal trademarks with their matches")
    processing_time_seconds: float = Field(default=0.0, description="Time taken to generate the report")
    
    @property
    def entries_with_similarities(self) -> List[JournalTrademarkEntry]:
        """Get only entries that have at least one similarity"""
        return [entry for entry in self.entries if entry.has_similarities]
    
    @property
    def entries_without_similarities(self) -> List[JournalTrademarkEntry]:
        """Get entries with no similarities"""
        return [entry for entry in self.entries if not entry.has_similarities]
    
    def get_high_similarity_entries(self, threshold: float = 0.8) -> List[JournalTrademarkEntry]:
        """Get entries with similarity scores above a threshold"""
        return [entry for entry in self.entries 
                if entry.has_similarities and entry.highest_similarity >= threshold]

