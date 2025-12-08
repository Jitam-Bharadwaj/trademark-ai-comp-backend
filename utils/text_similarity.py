"""
Text Similarity Utilities
Implements various NLP-based fuzzy matching algorithms for text comparison
"""
import logging
from typing import List, Tuple, Optional
import re

logger = logging.getLogger(__name__)

# Try to import optional libraries
try:
    from Levenshtein import distance as levenshtein_distance, ratio as levenshtein_ratio
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    logger.warning("python-Levenshtein not available. Using fallback implementation.")

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logger.warning("fuzzywuzzy not available. Some features will be limited.")

try:
    from phonetics import metaphone, soundex
    PHONETICS_AVAILABLE = True
except ImportError:
    PHONETICS_AVAILABLE = False
    logger.warning("phonetics library not available. Phonetic matching will be disabled.")


class TextSimilarity:
    """Calculate text similarity using multiple algorithms"""
    
    def __init__(self, 
                 use_levenshtein: bool = True,
                 use_phonetic: bool = True,
                 use_fuzzywuzzy: bool = True):
        """
        Initialize text similarity calculator
        
        Args:
            use_levenshtein: Use Levenshtein distance
            use_phonetic: Use phonetic matching (Soundex/Metaphone)
            use_fuzzywuzzy: Use fuzzywuzzy library
        """
        self.use_levenshtein = use_levenshtein and LEVENSHTEIN_AVAILABLE
        self.use_phonetic = use_phonetic and PHONETICS_AVAILABLE
        self.use_fuzzywuzzy = use_fuzzywuzzy and FUZZYWUZZY_AVAILABLE
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters (keep alphanumeric and spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Remove multiple spaces
        text = " ".join(text.split())
        
        return text.strip()
    
    def levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using Levenshtein distance
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        text1 = self.normalize_text(text1)
        text2 = self.normalize_text(text2)
        
        if text1 == text2:
            return 1.0
        
        if LEVENSHTEIN_AVAILABLE:
            # Use optimized Levenshtein library
            return levenshtein_ratio(text1, text2)
        else:
            # Fallback implementation
            return self._levenshtein_ratio_fallback(text1, text2)
    
    def _levenshtein_ratio_fallback(self, text1: str, text2: str) -> float:
        """Fallback Levenshtein ratio calculation"""
        if not text1 or not text2:
            return 0.0
        
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        
        distance = self._levenshtein_distance_fallback(text1, text2)
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance_fallback(self, text1: str, text2: str) -> int:
        """Calculate Levenshtein distance (fallback)"""
        if len(text1) < len(text2):
            return self._levenshtein_distance_fallback(text2, text1)
        
        if len(text2) == 0:
            return len(text1)
        
        previous_row = range(len(text2) + 1)
        for i, c1 in enumerate(text1):
            current_row = [i + 1]
            for j, c2 in enumerate(text2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def jaro_winkler_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaro-Winkler similarity
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        text1 = self.normalize_text(text1)
        text2 = self.normalize_text(text2)
        
        if text1 == text2:
            return 1.0
        
        if FUZZYWUZZY_AVAILABLE:
            return fuzz.WRatio(text1, text2) / 100.0
        
        # Fallback: Use Jaro similarity
        return self._jaro_similarity(text1, text2)
    
    def _jaro_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaro similarity"""
        if text1 == text2:
            return 1.0
        
        len1, len2 = len(text1), len(text2)
        match_window = max(len1, len2) // 2 - 1
        
        if match_window < 0:
            match_window = 0
        
        text1_matches = [False] * len1
        text2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len1):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len2)
            
            for j in range(start, end):
                if text2_matches[j] or text1[i] != text2[j]:
                    continue
                text1_matches[i] = True
                text2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Find transpositions
        k = 0
        for i in range(len1):
            if not text1_matches[i]:
                continue
            while not text2_matches[k]:
                k += 1
            if text1[i] != text2[k]:
                transpositions += 1
            k += 1
        
        jaro = (
            matches / len1 +
            matches / len2 +
            (matches - transpositions / 2) / matches
        ) / 3.0
        
        # Winkler modification
        prefix = 0
        for i in range(min(len(text1), len(text2), 4)):
            if text1[i] == text2[i]:
                prefix += 1
            else:
                break
        
        return jaro + (0.1 * prefix * (1 - jaro))
    
    def token_sort_ratio(self, text1: str, text2: str) -> float:
        """
        Calculate token-based similarity (word order independent)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        text1 = self.normalize_text(text1)
        text2 = self.normalize_text(text2)
        
        if FUZZYWUZZY_AVAILABLE:
            return fuzz.token_sort_ratio(text1, text2) / 100.0
        
        # Fallback: Sort tokens and compare
        tokens1 = sorted(text1.split())
        tokens2 = sorted(text2.split())
        return self.levenshtein_similarity(" ".join(tokens1), " ".join(tokens2))
    
    def phonetic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate phonetic similarity using Soundex/Metaphone
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not self.use_phonetic or not PHONETICS_AVAILABLE:
            return 0.0
        
        if not text1 or not text2:
            return 0.0
        
        text1 = self.normalize_text(text1)
        text2 = self.normalize_text(text2)
        
        try:
            # Use Metaphone (more accurate than Soundex)
            metaphone1 = metaphone(text1)
            metaphone2 = metaphone(text2)
            
            if metaphone1 == metaphone2:
                return 1.0
            
            # Also check Soundex
            soundex1 = soundex(text1)
            soundex2 = soundex(text2)
            
            if soundex1 == soundex2:
                return 0.8
            
            # Partial match on Metaphone
            if metaphone1 and metaphone2:
                if metaphone1.startswith(metaphone2[:2]) or metaphone2.startswith(metaphone1[:2]):
                    return 0.5
            
            return 0.0
        except Exception as e:
            logger.warning(f"Phonetic similarity error: {e}")
            return 0.0
    
    def calculate_similarity(self, text1: str, text2: str, 
                           weights: Optional[dict] = None) -> float:
        """
        Calculate combined similarity score using multiple algorithms
        
        Args:
            text1: First text
            text2: Second text
            weights: Dictionary with weights for each algorithm
                     Default: {'levenshtein': 0.4, 'jaro_winkler': 0.3, 
                              'token_sort': 0.2, 'phonetic': 0.1}
            
        Returns:
            Combined similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Default weights
        if weights is None:
            weights = {
                'levenshtein': 0.4,
                'jaro_winkler': 0.3,
                'token_sort': 0.2,
                'phonetic': 0.1
            }
        
        scores = {}
        
        # Calculate individual scores
        if self.use_levenshtein:
            scores['levenshtein'] = self.levenshtein_similarity(text1, text2)
        
        scores['jaro_winkler'] = self.jaro_winkler_similarity(text1, text2)
        scores['token_sort'] = self.token_sort_ratio(text1, text2)
        
        if self.use_phonetic:
            scores['phonetic'] = self.phonetic_similarity(text1, text2)
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for algorithm, weight in weights.items():
            if algorithm in scores:
                weighted_sum += scores[algorithm] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        final_score = weighted_sum / total_weight
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, final_score))
    
    def find_best_matches(self, query_text: str, candidate_texts: List[str],
                         top_k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Find best matching texts from a list of candidates
        
        Args:
            query_text: Query text to match
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (text, similarity_score) tuples, sorted by score descending
        """
        if not query_text or not candidate_texts:
            return []
        
        matches = []
        
        for candidate in candidate_texts:
            if not candidate:
                continue
            
            similarity = self.calculate_similarity(query_text, candidate)
            
            if similarity >= threshold:
                matches.append((candidate, similarity))
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return matches[:top_k]







