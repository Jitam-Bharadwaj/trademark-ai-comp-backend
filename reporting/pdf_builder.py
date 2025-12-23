"""
PDF Report Builder for Weekly Trademark Similarity Reports

Uses reportlab to generate formatted PDF reports.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from config import Config
from reporting.report_models import WeeklyReport, JournalTrademarkEntry, SimilarTrademark
from reporting.report_generator import classes_match

logger = logging.getLogger(__name__)


class PDFReportBuilder:
    """Builds PDF reports from WeeklyReport data"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the PDF builder
        
        Args:
            output_dir: Directory to save PDF reports. Defaults to Config.REPORT_OUTPUT_PATH
        """
        self.output_dir = output_dir or Config.REPORT_OUTPUT_PATH
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup styles
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a365d')
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a5568')
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2d3748')
        ))
        
        # Entry header (for each trademark)
        self.styles.add(ParagraphStyle(
            name='EntryHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=5,
            textColor=colors.HexColor('#2b6cb0')
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='ReportBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.HexColor('#1a202c')
        ))
        
        # Small text for details
        self.styles.add(ParagraphStyle(
            name='ReportDetailText',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#4a5568')
        ))
        
        # High similarity warning
        self.styles.add(ParagraphStyle(
            name='HighSimilarity',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#c53030'),
            backColor=colors.HexColor('#fed7d7')
        ))
    
    def build_report(self, report: WeeklyReport) -> Path:
        """
        Build a PDF report from WeeklyReport data
        
        Args:
            report: WeeklyReport object containing all data
            
        Returns:
            Path to the generated PDF file
        """
        # Generate filename
        filename = f"{report.report_id}.pdf"
        filepath = self.output_dir / filename
        
        logger.info(f"Building PDF report: {filepath}")
        
        # Create document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=1*cm,
            leftMargin=1*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm
        )
        
        # Build content
        story = []
        
        # Add cover page
        story.extend(self._build_cover_page(report))
        
        # Add summary section
        story.extend(self._build_summary_section(report))
        
        # Add detailed entries section
        story.extend(self._build_entries_section(report))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {filepath}")
        return filepath
    
    def _build_cover_page(self, report: WeeklyReport) -> list:
        """Build the cover page content"""
        elements = []
        
        # Add some space at the top
        elements.append(Spacer(1, 2*inch))
        
        # Title
        elements.append(Paragraph(
            "Weekly Trademark Similarity Report",
            self.styles['ReportTitle']
        ))
        
        # Subtitle with dates
        elements.append(Paragraph(
            f"Journal Upload Date: {report.monday_date.strftime('%A, %B %d, %Y')}",
            self.styles['ReportSubtitle']
        ))
        
        elements.append(Paragraph(
            f"Report Generated: {report.report_date.strftime('%B %d, %Y at %H:%M')}",
            self.styles['ReportSubtitle']
        ))
        
        elements.append(Spacer(1, 1*inch))
        
        # Quick stats box
        quick_stats = [
            ['Quick Statistics', ''],
            ['Journal Trademarks Analyzed', str(report.summary.total_journal_trademarks)],
            ['Image-Based', str(report.summary.total_image_based)],
            ['Text-Only', str(report.summary.total_text_only)],
            ['Self Database Trademarks', str(report.summary.total_self_db_trademarks)],
            ['Similarities Found', str(report.summary.total_similarities_found)],
            ['Trademarks with Matches', str(report.summary.trademarks_with_similarities)],
        ]
        
        stats_table = Table(quick_stats, colWidths=[3.5*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('SPAN', (0, 0), (1, 0)),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ]))
        
        elements.append(stats_table)
        
        elements.append(PageBreak())
        
        return elements
    
    def _build_summary_section(self, report: WeeklyReport) -> list:
        """Build the summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 0.3*inch))
        
        summary = report.summary
        
        # Summary text
        summary_text = f"""
        This report analyzes <b>{summary.total_journal_trademarks}</b> trademarks from the Indian Trademark 
        Journal uploaded on <b>{report.monday_date.strftime('%B %d, %Y')}</b>. 
        Of these, <b>{summary.total_image_based}</b> are image-based trademarks and 
        <b>{summary.total_text_only}</b> are text-only trademarks.
        <br/><br/>
        These journal trademarks were compared against <b>{summary.total_self_db_trademarks}</b> 
        trademarks in the self database using a similarity threshold of <b>{summary.similarity_threshold_used}</b>.
        <br/><br/>
        <b>Key Findings:</b>
        <br/>
        • Total similarity matches found: <b>{summary.total_similarities_found}</b>
        <br/>
        • Journal trademarks with at least one match: <b>{summary.trademarks_with_similarities}</b> 
        ({self._calc_percentage(summary.trademarks_with_similarities, summary.total_journal_trademarks)}%)
        <br/>
        • Highest similarity score: <b>{summary.highest_similarity_score:.2%}</b>
        <br/>
        • Average similarity score: <b>{summary.average_similarity_score:.2%}</b>
        """
        
        elements.append(Paragraph(summary_text, self.styles['ReportBodyText']))
        
        # High similarity warning if applicable
        high_sim_entries = report.get_high_similarity_entries(threshold=0.8)
        if high_sim_entries:
            elements.append(Spacer(1, 0.3*inch))
            warning_text = f"""
            <b>⚠️ Attention:</b> {len(high_sim_entries)} trademark(s) have similarity scores 
            of 80% or higher. These require immediate review.
            """
            elements.append(Paragraph(warning_text, self.styles['HighSimilarity']))
        
        elements.append(Spacer(1, 0.5*inch))
        
        return elements
    
    def _build_entries_section(self, report: WeeklyReport) -> list:
        """Build the detailed entries section"""
        elements = []
        
        # Minimum similarity threshold for display (30%)
        MIN_DISPLAY_THRESHOLD = 0.30
        
        elements.append(Paragraph("Detailed Similarity Analysis", self.styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 0.3*inch))
        
        # Get entries with similarities >= 30% (sorted by highest match score)
        # Filter to only include entries that have at least one match >= 30%
        entries_with_matches = []
        for entry in report.entries_with_similarities:
            filtered_similar = [s for s in entry.similar_trademarks if s.similarity_score >= MIN_DISPLAY_THRESHOLD]
            if filtered_similar:
                # Calculate highest score from filtered matches
                highest_filtered = max(s.similarity_score for s in filtered_similar)
                entries_with_matches.append((entry, highest_filtered))
        
        # Sort by highest filtered similarity score
        entries_with_matches.sort(key=lambda x: x[1], reverse=True)
        
        if not entries_with_matches:
            elements.append(Paragraph(
                "No similarities found above 30% threshold.",
                self.styles['ReportBodyText']
            ))
            return elements
        
        elements.append(Paragraph(
            f"Showing {len(entries_with_matches)} journal trademark(s) with similarity matches (≥30%):",
            self.styles['ReportBodyText']
        ))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add page break before starting entries
        elements.append(PageBreak())
        
        # Add each entry - ONE ENTRY PER PAGE
        for idx, (entry, _) in enumerate(entries_with_matches, 1):
            elements.extend(self._build_single_entry(entry, idx))
            
            # Add page break after each entry (except the last one)
            if idx < len(entries_with_matches):
                elements.append(PageBreak())
        
        return elements
    
    def _build_single_entry(self, entry: JournalTrademarkEntry, index: int) -> list:
        """Build content for a single journal trademark entry"""
        elements = []
        
        # Entry header
        type_badge = "📷 Image" if entry.trademark_type == 'image_based' else "📝 Text"
        header_text = f"{index}. {entry.mark_name or 'Unknown Mark'} [{type_badge}]"
        elements.append(Paragraph(header_text, self.styles['EntryHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        # For image-based trademarks, show the image
        if entry.trademark_type == 'image_based' and entry.image_path:
            image_element = self._build_trademark_image(entry.image_path)
            if image_element:
                elements.append(image_element)
                elements.append(Spacer(1, 0.15*inch))
        
        # Filter similar trademarks to only include scores >= 30%
        filtered_similar = [sim for sim in entry.similar_trademarks if sim.similarity_score >= 0.30]
        
        # Journal trademark details - FULL WIDTH (7 inches total)
        details = [
            ['Journal Trademark Details', ''],
            ['Type', entry.trademark_type.replace('_', ' ').title()],
            ['Mark Name', entry.mark_name or 'N/A'],
            ['Class', entry.trademark_class or 'N/A'],
            ['Applicant', entry.applicant_name or 'N/A'],
            ['Application No.', entry.application_no or 'N/A'],
            ['PDF Source', entry.pdf_source or 'N/A'],
            ['Page', str(entry.page_number)],
        ]
        
        details_table = Table(details, colWidths=[2*inch, 5*inch])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
            ('SPAN', (0, 0), (1, 0)),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ]))
        
        elements.append(details_table)
        elements.append(Spacer(1, 0.1*inch))
        
        # Similar trademarks table - only show if there are matches >= 30%
        if not filtered_similar:
            elements.append(Paragraph(
                "No similar trademarks found above 30% threshold.",
                self.styles['ReportDetailText']
            ))
        else:
            elements.append(Paragraph(
                f"Similar Trademarks Found ({len(filtered_similar)}):",
                self.styles['ReportDetailText']
            ))
            
            # Build similar trademarks table - FULL WIDTH (7 inches total)
            sim_headers = ['Mark', 'Score', 'Class', 'Applicant', 'App. No.']
            sim_data = [sim_headers]
            
            for sim in filtered_similar:
                # Truncate long text
                mark = self._truncate(sim.mark, 30)
                applicant = self._truncate(sim.applicant_name, 25)
                
                # Format score with color indicator
                score_str = f"{sim.similarity_score:.1%}"
                
                sim_data.append([
                    mark,
                    score_str,
                    sim.trademark_class or 'N/A',
                    applicant or 'N/A',
                    sim.application_no or 'N/A'
                ])
            
            # FULL WIDTH table: 2 + 0.8 + 0.7 + 2.2 + 1.3 = 7 inches
            sim_table = Table(sim_data, colWidths=[2*inch, 0.8*inch, 0.7*inch, 2.2*inch, 1.3*inch])
            
            # Table styling
            table_style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ]
            
            # Highlight rows based on class match and similarity score
            # Same-class matches get priority highlighting
            journal_class = entry.trademark_class or ''
            for row_idx, sim in enumerate(filtered_similar, 1):
                # Use classes_match() to handle different class formats (e.g., "5" vs "Class 5")
                is_same_class = classes_match(sim.trademark_class, journal_class)
                
                if is_same_class:
                    # Same class matches: use distinct highlighting
                    if sim.similarity_score >= 0.8:
                        # High similarity + same class: strong red highlight
                        table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#fed7d7')))
                    elif sim.similarity_score >= 0.7:
                        # Medium similarity + same class: orange highlight
                        table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#feebc8')))
                    else:
                        # Lower similarity + same class: light green highlight to indicate class match
                        table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#c6f6d5')))
                else:
                    # Different class: standard highlighting based on score only
                    if sim.similarity_score >= 0.8:
                        table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#fed7d7')))
                    elif sim.similarity_score >= 0.7:
                        table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#feebc8')))
                    else:
                        table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.white))
            
            sim_table.setStyle(TableStyle(table_style))
            elements.append(sim_table)
        
        elements.append(Spacer(1, 0.1*inch))
        
        # Add similarity summary in words (using filtered similar trademarks)
        summary_elements = self._build_similarity_summary(entry, filtered_similar)
        elements.extend(summary_elements)
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _calc_percentage(self, part: int, total: int) -> str:
        """Calculate percentage as string"""
        if total == 0:
            return "0"
        return f"{(part / total) * 100:.1f}"
    
    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis"""
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."
    
    def _build_similarity_summary(self, entry: JournalTrademarkEntry, 
                                    filtered_similar: list = None) -> list:
        """
        Build a human-readable summary of the similarities found
        
        Args:
            entry: The journal trademark entry with its similar trademarks
            filtered_similar: Optional pre-filtered list of similar trademarks (>= 30%)
            
        Returns:
            List of reportlab elements containing the summary
        """
        elements = []
        
        # Use filtered list if provided, otherwise use entry's similar_trademarks
        similar_marks = filtered_similar if filtered_similar is not None else entry.similar_trademarks
        
        if not similar_marks:
            return elements
        
        # Build summary text
        summary_lines = []
        
        # Overall summary
        journal_mark = entry.mark_name or "Unknown Mark"
        tm_type = "image-based" if entry.trademark_type == 'image_based' else "text-based"
        num_matches = len(similar_marks)
        highest_score = max(s.similarity_score for s in similar_marks) if similar_marks else 0
        
        # Introduction
        intro = f"The journal trademark <b>\"{journal_mark}\"</b> ({tm_type}) was compared against the self-database trademarks."
        summary_lines.append(intro)
        
        # Match overview
        if num_matches == 1:
            match_text = f"<b>1 similar trademark</b> was found with a similarity score of <b>{highest_score:.1%}</b>."
        else:
            match_text = f"<b>{num_matches} similar trademarks</b> were found, with the highest similarity score being <b>{highest_score:.1%}</b>."
        summary_lines.append(match_text)
        
        # Detailed analysis for each match - show ALL matches
        for idx, sim in enumerate(similar_marks, 1):
            match_details = self._generate_match_description(entry, sim, idx)
            summary_lines.append(match_details)
        
        # Risk assessment
        risk_assessment = self._generate_risk_assessment_from_list(similar_marks, highest_score)
        if risk_assessment:
            summary_lines.append(risk_assessment)
        
        # Create styled paragraphs that can flow across pages naturally
        # Use Paragraphs directly (not wrapped in Table) for perfect page flow
        
        # Create styled paragraphs that can flow across pages naturally
        # Use a Table with multiple rows - each row can split across pages
        
        # Build table data: header + content rows
        table_data = []
        
        # Header row
        header_para = Paragraph("<b>📋 Similarity Analysis Summary</b>", self.styles['ReportDetailText'])
        table_data.append([header_para])
        
        # Content rows - each summary line is its own row for better page flow
        for line in summary_lines:
            content_para = Paragraph(line, self.styles['ReportBodyText'])
            table_data.append([content_para])
        
        # Create table with styling - this will split across pages naturally
        summary_table = Table(
            table_data,
            colWidths=[7*inch]
        )
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f9ff')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bee3f8')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            # Allow rows to split across pages
            ('INNERGRID', (0, 0), (-1, -1), 0, colors.transparent),
        ]))
        
        elements.append(summary_table)
        
        return elements
    
    def _generate_risk_assessment_from_list(self, similar_marks: list, highest_score: float) -> str:
        """
        Generate a risk assessment based on similarity scores from a list
        
        Args:
            similar_marks: List of similar trademarks
            highest_score: Highest similarity score
            
        Returns:
            Risk assessment text
        """
        if not similar_marks:
            return ""
        
        num_high = sum(1 for s in similar_marks if s.similarity_score >= 0.8)
        num_medium = sum(1 for s in similar_marks if 0.5 <= s.similarity_score < 0.8)
        
        if highest_score >= 0.8:
            risk_level = "🔴 <b>HIGH RISK</b>"
            risk_desc = f"Very high similarity ({highest_score:.1%}). Immediate review recommended. {num_high} match(es) above 80%."
        elif highest_score >= 0.6:
            risk_level = "🟡 <b>MEDIUM RISK</b>"
            risk_desc = f"Moderate similarity ({highest_score:.1%}). Further evaluation warranted. {num_medium} match(es) between 50-80%."
        else:
            risk_level = "🟢 <b>LOW RISK</b>"
            risk_desc = f"Lower similarity (highest: {highest_score:.1%}). Monitoring advised."
        
        return f"<b>Risk Assessment:</b> {risk_level} — {risk_desc}"
    
    def _generate_match_description(self, entry: JournalTrademarkEntry, 
                                     sim: 'SimilarTrademark', match_num: int) -> str:
        """
        Generate a detailed description of a specific match
        
        Args:
            entry: The journal trademark entry
            sim: The similar trademark from self-database
            match_num: The match number (1, 2, 3...)
            
        Returns:
            HTML-formatted description string
        """
        journal_mark = entry.mark_name or "Unknown"
        self_mark = sim.mark or "Unknown"
        score = sim.similarity_score
        
        # Determine similarity type
        has_vector = sim.vector_similarity_score is not None and sim.vector_similarity_score > 0
        has_text = sim.text_similarity_score is not None and sim.text_similarity_score > 0
        
        # Build description
        desc_parts = [f"<b>Match {match_num}:</b> \"{self_mark}\""]
        
        # Similarity type explanation
        if entry.trademark_type == 'image_based':
            if has_vector:
                desc_parts.append(f"Visual/image similarity detected ({sim.vector_similarity_score:.1%} visual match)")
                if has_text:
                    desc_parts.append(f"with additional text similarity ({sim.text_similarity_score:.1%})")
            elif has_text:
                desc_parts.append(f"Text similarity detected ({sim.text_similarity_score:.1%}) based on extracted text/OCR")
            else:
                desc_parts.append(f"Cross-modal similarity detected ({score:.1%})")
        else:
            # Text-based trademark
            if has_text:
                desc_parts.append(f"Text similarity detected ({sim.text_similarity_score:.1%})")
                # Explain text similarity aspects
                text_analysis = self._analyze_text_similarity(journal_mark, self_mark)
                if text_analysis:
                    desc_parts.append(text_analysis)
            else:
                desc_parts.append(f"Similarity score: {score:.1%}")
        
        # Class match information
        if entry.trademark_class and sim.trademark_class:
            # Use classes_match() to handle different class formats (e.g., "5" vs "Class 5")
            if classes_match(entry.trademark_class, sim.trademark_class):
                desc_parts.append(f"<i>Both marks are in Class {sim.trademark_class} (same class - higher conflict risk)</i>")
            else:
                desc_parts.append(f"<i>Self DB mark is in Class {sim.trademark_class}</i>")
        
        return " — ".join(desc_parts)
    
    def _analyze_text_similarity(self, journal_mark: str, self_mark: str) -> str:
        """
        Analyze and describe the nature of text similarity
        
        Args:
            journal_mark: The journal trademark text
            self_mark: The self-database trademark text
            
        Returns:
            Description of similarity aspects
        """
        if not journal_mark or not self_mark or journal_mark == 'N/A':
            return ""
        
        j_lower = journal_mark.lower().strip()
        s_lower = self_mark.lower().strip()
        
        aspects = []
        
        # Exact match
        if j_lower == s_lower:
            return "Exact text match"
        
        # Prefix/suffix match
        if j_lower.startswith(s_lower) or s_lower.startswith(j_lower):
            aspects.append("shares common prefix")
        elif j_lower.endswith(s_lower) or s_lower.endswith(j_lower):
            aspects.append("shares common suffix")
        
        # Contains match
        if s_lower in j_lower or j_lower in s_lower:
            aspects.append("one mark contains the other")
        
        # Common words
        j_words = set(j_lower.split())
        s_words = set(s_lower.split())
        common_words = j_words & s_words
        if common_words:
            aspects.append(f"common words: {', '.join(common_words)}")
        
        # Similar length
        len_diff = abs(len(journal_mark) - len(self_mark))
        if len_diff <= 2:
            aspects.append("similar length")
        
        if aspects:
            return f"({'; '.join(aspects)})"
        return ""
    

    def _build_trademark_image(self, image_path: str, max_width: float = 2.5, 
                                max_height: float = 2.0):
        """
        Build an image element for the trademark image
        
        Args:
            image_path: Path to the image file
            max_width: Maximum width in inches
            max_height: Maximum height in inches
            
        Returns:
            Table element containing image or None if image cannot be loaded
        """
        try:
            from PIL import Image as PILImage
            import os
            
            # Check if file exists
            if not image_path or not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
            
            # Open image to get dimensions
            with PILImage.open(image_path) as img:
                orig_width, orig_height = img.size
            
            # Convert max dimensions to points (1 inch = 72 points)
            max_width_pts = max_width * inch
            max_height_pts = max_height * inch
            
            # Calculate scaling to fit within max dimensions while maintaining aspect ratio
            width_ratio = max_width_pts / orig_width
            height_ratio = max_height_pts / orig_height
            scale = min(width_ratio, height_ratio, 1.0)  # Don't upscale
            
            final_width = orig_width * scale
            final_height = orig_height * scale
            
            # Create reportlab image
            rl_image = RLImage(image_path, width=final_width, height=final_height)
            
            # Wrap in a table for centering and border
            image_table = Table(
                [[rl_image]],
                colWidths=[final_width + 10],
                rowHeights=[final_height + 10]
            )
            image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7fafc')),
            ]))
            
            return image_table
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

