"""
Database Query Module for Trademark Applications
Queries the tr_applications table to get mark fields for text similarity matching
"""
import logging
from typing import List, Dict, Optional
from database.db_connection import db

logger = logging.getLogger(__name__)


class ApplicationQueries:
    """Query trademark applications from MySQL database"""
    
    def __init__(self):
        """Initialize application queries"""
        self._marks_cache: Optional[List[Dict]] = None
        self._cache_enabled = True
    
    def get_all_marks(self, use_cache: bool = True) -> List[Dict]:
        """
        Get all mark fields from tr_applications table
        
        Args:
            use_cache: Whether to use cached results
            
        Returns:
            List of dictionaries with 'application_id' and 'mark' fields
            Example: [{'application_id': 1, 'mark': 'VFS in Circle'}, ...]
        """
        if use_cache and self._cache_enabled and self._marks_cache is not None:
            logger.debug("Returning cached marks")
            return self._marks_cache
        
        try:
            query = """
                SELECT id as application_id, mark 
                FROM tr_applications 
                WHERE mark IS NOT NULL 
                AND mark != '' 
                AND mark != 'NULL'
                ORDER BY id
            """
            
            results = db.execute_query(query, fetch_all=True)
            
            if results is None:
                logger.warning("No marks found in database")
                return []
            
            # Convert to list of dicts if needed
            marks_list = []
            for row in results:
                if isinstance(row, dict):
                    marks_list.append({
                        'application_id': row.get('application_id'),
                        'mark': row.get('mark', '')
                    })
                else:
                    # Handle tuple results
                    marks_list.append({
                        'application_id': row[0],
                        'mark': row[1] if len(row) > 1 else ''
                    })
            
            # Cache results
            if use_cache and self._cache_enabled:
                self._marks_cache = marks_list
                logger.info(f"Cached {len(marks_list)} marks from database")
            
            logger.info(f"Retrieved {len(marks_list)} marks from database")
            return marks_list
            
        except Exception as e:
            logger.error(f"Error querying marks from database: {e}")
            return []
    
    def get_marks_by_client(self, client_id: int) -> List[Dict]:
        """
        Get marks for a specific client
        
        Args:
            client_id: Client ID
            
        Returns:
            List of dictionaries with 'application_id' and 'mark' fields
        """
        try:
            query = """
                SELECT id as application_id, mark 
                FROM tr_applications 
                WHERE client_id = %s 
                AND mark IS NOT NULL 
                AND mark != '' 
                AND mark != 'NULL'
                ORDER BY id
            """
            
            results = db.execute_query(query, params=(client_id,), fetch_all=True)
            
            if results is None:
                return []
            
            marks_list = []
            for row in results:
                if isinstance(row, dict):
                    marks_list.append({
                        'application_id': row.get('application_id'),
                        'mark': row.get('mark', '')
                    })
                else:
                    marks_list.append({
                        'application_id': row[0],
                        'mark': row[1] if len(row) > 1 else ''
                    })
            
            logger.info(f"Retrieved {len(marks_list)} marks for client {client_id}")
            return marks_list
            
        except Exception as e:
            logger.error(f"Error querying marks for client {client_id}: {e}")
            return []
    
    def get_application_by_id(self, application_id: int) -> Optional[Dict]:
        """
        Get application details by ID
        
        Args:
            application_id: Application ID
            
        Returns:
            Dictionary with application details or None
        """
        try:
            query = """
                SELECT id, client_id, mark, type_of_trademark, 
                       application_type, country_id, created_at
                FROM tr_applications 
                WHERE id = %s
            """
            
            result = db.execute_query(query, params=(application_id,), fetch_one=True)
            
            if result is None:
                return None
            
            if isinstance(result, dict):
                return result
            else:
                # Convert tuple to dict
                return {
                    'id': result[0],
                    'client_id': result[1] if len(result) > 1 else None,
                    'mark': result[2] if len(result) > 2 else None,
                    'type_of_trademark': result[3] if len(result) > 3 else None,
                    'application_type': result[4] if len(result) > 4 else None,
                    'country_id': result[5] if len(result) > 5 else None,
                    'created_at': result[6] if len(result) > 6 else None
                }
                
        except Exception as e:
            logger.error(f"Error querying application {application_id}: {e}")
            return None
    
    def get_mark_by_application_id(self, application_id: int) -> Optional[str]:
        """
        Get mark field for a specific application
        
        Args:
            application_id: Application ID
            
        Returns:
            Mark string or None
        """
        try:
            query = """
                SELECT mark 
                FROM tr_applications 
                WHERE id = %s
            """
            
            result = db.execute_query(query, params=(application_id,), fetch_one=True)
            
            if result is None:
                return None
            
            if isinstance(result, dict):
                return result.get('mark')
            else:
                return result[0] if len(result) > 0 else None
                
        except Exception as e:
            logger.error(f"Error querying mark for application {application_id}: {e}")
            return None
    
    def get_applications_by_ids(self, application_ids: List[int]) -> Dict[int, Dict]:
        """
        Get application details including class for multiple application IDs
        
        Args:
            application_ids: List of application IDs
            
        Returns:
            Dictionary mapping application_id to application details
            Example: {1: {'id': 1, 'mark': 'ABC', 'trademark_class': '25', ...}, ...}
        """
        if not application_ids:
            return {}
        
        try:
            # Create placeholders for IN clause
            placeholders = ','.join(['%s'] * len(application_ids))
            # Query to get application details including trademark class, applicant name, and application_no
            # Join with tr_application_classes, tr_classes, tr_application_applicants, tr_users, and tr_application_details
            query = f"""
                SELECT 
                    ta.id,
                    ta.mark,
                    ta.type_of_trademark,
                    ta.application_type,
                    ta.country_id,
                    ta.client_id,
                    ta.created_at,
                    GROUP_CONCAT(DISTINCT tc.class_name ORDER BY tc.id SEPARATOR ', ') as trademark_class,
                    GROUP_CONCAT(DISTINCT tc.id ORDER BY tc.id SEPARATOR ', ') as trademark_class_ids,
                    GROUP_CONCAT(DISTINCT CONCAT(COALESCE(tu.first_name, ''), ' ', COALESCE(tu.last_name, '')) 
                                ORDER BY tu.id SEPARATOR ', ') as applicant_name,
                    MAX(tad.application_no) as application_no
                FROM tr_applications ta
                LEFT JOIN tr_application_classes tac ON ta.id = tac.application_id
                LEFT JOIN tr_classes tc ON tac.class_id = tc.id
                LEFT JOIN tr_application_applicants taa ON ta.id = taa.application_id
                LEFT JOIN tr_users tu ON taa.applicant_id = tu.id
                LEFT JOIN tr_application_details tad ON ta.id = tad.application_id
                WHERE ta.id IN ({placeholders})
                GROUP BY ta.id, ta.mark, ta.type_of_trademark, ta.application_type, 
                         ta.country_id, ta.client_id, ta.created_at
            """
            
            results = db.execute_query(query, params=tuple(application_ids), fetch_all=True)
            
            if results is None:
                return {}
            
            applications_dict = {}
            for row in results:
                if isinstance(row, dict):
                    app_id = row.get('id')
                    if app_id:
                        # Get trademark_class from the JOIN query result
                        # It will be a comma-separated string of class names like "Class 1, Class 2"
                        trademark_class = row.get('trademark_class') or ''
                        # Clean up the class string (remove extra spaces, handle NULL)
                        if trademark_class:
                            trademark_class = ', '.join([c.strip() for c in trademark_class.split(',') if c.strip()])
                        
                        # Get applicant_name from the JOIN query result
                        # It will be a comma-separated string of applicant names
                        applicant_name = row.get('applicant_name') or ''
                        # Clean up the applicant name string (remove extra spaces, handle NULL)
                        if applicant_name:
                            # Remove extra spaces and clean up
                            applicant_name = ', '.join([name.strip() for name in applicant_name.split(',') if name.strip()])
                            # Remove any double spaces that might occur
                            applicant_name = ' '.join(applicant_name.split())
                        
                        # Get application_no from the JOIN query result
                        application_no = row.get('application_no') or ''
                        
                        applications_dict[app_id] = {
                            'id': app_id,
                            'mark': row.get('mark'),
                            'trademark_class': trademark_class,
                            'applicant_name': applicant_name,
                            'application_no': application_no,
                            'type_of_trademark': row.get('type_of_trademark'),
                            'application_type': row.get('application_type'),
                            'country_id': row.get('country_id'),
                            'client_id': row.get('client_id'),
                            'created_at': row.get('created_at')
                        }
                else:
                    # Handle tuple results
                    # Order: id, mark, type_of_trademark, application_type, country_id, 
                    #        client_id, created_at, trademark_class, trademark_class_ids, applicant_name
                    app_id = row[0] if len(row) > 0 else None
                    if app_id:
                        # Extract trademark_class from tuple (at index 7)
                        trademark_class = ''
                        if len(row) > 7:
                            trademark_class = row[7] or ''
                            if trademark_class:
                                trademark_class = ', '.join([c.strip() for c in trademark_class.split(',') if c.strip()])
                        
                        # Extract applicant_name from tuple (at index 9)
                        applicant_name = ''
                        if len(row) > 9:
                            applicant_name = row[9] or ''
                            if applicant_name:
                                applicant_name = ', '.join([name.strip() for name in applicant_name.split(',') if name.strip()])
                                applicant_name = ' '.join(applicant_name.split())  # Remove double spaces
                        
                        # Extract application_no from tuple (at index 10)
                        application_no = ''
                        if len(row) > 10:
                            application_no = row[10] or ''
                        
                        applications_dict[app_id] = {
                            'id': app_id,
                            'mark': row[1] if len(row) > 1 else None,
                            'trademark_class': trademark_class,
                            'applicant_name': applicant_name,
                            'application_no': application_no,
                            'type_of_trademark': row[2] if len(row) > 2 else None,
                            'application_type': row[3] if len(row) > 3 else None,
                            'country_id': row[4] if len(row) > 4 else None,
                            'client_id': row[5] if len(row) > 5 else None,
                            'created_at': row[6] if len(row) > 6 else None
                        }
            
            logger.debug(f"Retrieved {len(applications_dict)} applications with details")
            return applications_dict
                
        except Exception as e:
            logger.error(f"Error querying applications by IDs: {e}")
            # Fallback: try a simpler query without GROUP_CONCAT (in case of MySQL version issues)
            try:
                logger.warning("Retrying query with simpler JOIN...")
                placeholders = ','.join(['%s'] * len(application_ids))
                # Simpler query that gets one class, one applicant, and application_no per application (first one found)
                query = f"""
                    SELECT DISTINCT
                        ta.id,
                        ta.mark,
                        ta.type_of_trademark,
                        ta.application_type,
                        ta.country_id,
                        ta.client_id,
                        ta.created_at,
                        tc.class_name as trademark_class,
                        CONCAT(COALESCE(tu.first_name, ''), ' ', COALESCE(tu.last_name, '')) as applicant_name,
                        tad.application_no
                    FROM tr_applications ta
                    LEFT JOIN tr_application_classes tac ON ta.id = tac.application_id
                    LEFT JOIN tr_classes tc ON tac.class_id = tc.id
                    LEFT JOIN tr_application_applicants taa ON ta.id = taa.application_id
                    LEFT JOIN tr_users tu ON taa.applicant_id = tu.id
                    LEFT JOIN tr_application_details tad ON ta.id = tad.application_id
                    WHERE ta.id IN ({placeholders})
                """
                results = db.execute_query(query, params=tuple(application_ids), fetch_all=True)
                if results:
                    applications_dict = {}
                    # Group by application_id to collect all classes
                    for row in results:
                        if isinstance(row, dict):
                            app_id = row.get('id')
                            if app_id:
                                if app_id not in applications_dict:
                                    applications_dict[app_id] = {
                                        'id': app_id,
                                        'mark': row.get('mark'),
                                        'trademark_class': '',
                                        'applicant_name': '',
                                        'application_no': '',
                                        'type_of_trademark': row.get('type_of_trademark'),
                                        'application_type': row.get('application_type'),
                                        'country_id': row.get('country_id'),
                                        'client_id': row.get('client_id'),
                                        'created_at': row.get('created_at'),
                                        '_classes': [],  # Temporary list to collect classes
                                        '_applicants': [],  # Temporary list to collect applicants
                                        '_application_nos': []  # Temporary list to collect application numbers
                                    }
                                # Collect class names
                                class_name = row.get('trademark_class')
                                if class_name and class_name not in applications_dict[app_id]['_classes']:
                                    applications_dict[app_id]['_classes'].append(class_name)
                                
                                # Collect applicant names
                                applicant_name = row.get('applicant_name')
                                if applicant_name:
                                    applicant_name = applicant_name.strip()
                                    if applicant_name and applicant_name not in applications_dict[app_id]['_applicants']:
                                        applications_dict[app_id]['_applicants'].append(applicant_name)
                                
                                # Collect application numbers
                                application_no = row.get('application_no')
                                if application_no:
                                    application_no = application_no.strip()
                                    if application_no and application_no not in applications_dict[app_id]['_application_nos']:
                                        applications_dict[app_id]['_application_nos'].append(application_no)
                    
                    # Convert class, applicant, and application_no lists to comma-separated strings
                    for app_id, app_data in applications_dict.items():
                        if app_data.get('_classes'):
                            app_data['trademark_class'] = ', '.join(app_data['_classes'])
                        if app_data.get('_applicants'):
                            app_data['applicant_name'] = ', '.join(app_data['_applicants'])
                        if app_data.get('_application_nos'):
                            # Usually there's only one application_no, but handle multiple if they exist
                            app_data['application_no'] = ', '.join(app_data['_application_nos'])
                        del app_data['_classes']
                        del app_data['_applicants']
                        del app_data['_application_nos']
                    
                    return applications_dict
            except Exception as e2:
                logger.error(f"Error in fallback query: {e2}")
            return {}
    
    def search_marks_by_text(self, query_text: str, threshold: float = 0.0,
                            limit: int = 100) -> List[Dict]:
        """
        Search marks using SQL LIKE (basic text search)
        Note: For fuzzy matching, use TextSimilarity class with get_all_marks()
        
        Args:
            query_text: Text to search for
            threshold: Not used in SQL search (kept for API compatibility)
            limit: Maximum number of results
            
        Returns:
            List of matching applications
        """
        try:
            # Normalize query text
            query_text = query_text.strip()
            if not query_text:
                return []
            
            # Use LIKE for basic matching
            query = """
                SELECT id as application_id, mark 
                FROM tr_applications 
                WHERE mark IS NOT NULL 
                AND mark != '' 
                AND mark != 'NULL'
                AND (mark LIKE %s OR mark LIKE %s)
                ORDER BY id
                LIMIT %s
            """
            
            search_pattern = f"%{query_text}%"
            search_pattern_lower = f"%{query_text.lower()}%"
            
            results = db.execute_query(
                query, 
                params=(search_pattern, search_pattern_lower, limit), 
                fetch_all=True
            )
            
            if results is None:
                return []
            
            marks_list = []
            for row in results:
                if isinstance(row, dict):
                    marks_list.append({
                        'application_id': row.get('application_id'),
                        'mark': row.get('mark', '')
                    })
                else:
                    marks_list.append({
                        'application_id': row[0],
                        'mark': row[1] if len(row) > 1 else ''
                    })
            
            logger.info(f"SQL search found {len(marks_list)} marks matching '{query_text}'")
            return marks_list
            
        except Exception as e:
            logger.error(f"Error searching marks: {e}")
            return []
    
    def clear_cache(self):
        """Clear the marks cache"""
        self._marks_cache = None
        logger.info("Marks cache cleared")
    
    def refresh_cache(self):
        """Refresh the marks cache"""
        self.clear_cache()
        self.get_all_marks(use_cache=True)
    
    def disable_cache(self):
        """Disable caching"""
        self._cache_enabled = False
        self.clear_cache()
    
    def enable_cache(self):
        """Enable caching"""
        self._cache_enabled = True


# Global instance
application_queries = ApplicationQueries()



