"""
User Model for authentication
Implements methods for user management based on documentation
"""
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from database.db_connection import db
import logging

logger = logging.getLogger(__name__)


# Role constants as per documentation
ROLE_ADMIN = 'admin'
ROLE_EMPLOYEE = 'employee'
ROLE_CLIENTS = 'clients'
ROLE_APPLICANT = 'applicant'


class UserModel:
    """User model for handling user operations"""
    
    # Role constants
    ROLE_ADMIN = ROLE_ADMIN
    ROLE_EMPLOYEE = ROLE_EMPLOYEE
    ROLE_CLIENTS = ROLE_CLIENTS
    ROLE_APPLICANT = ROLE_APPLICANT
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using MD5 as per documentation
        
        Note: MD5 is insecure. This is maintained for compatibility
        with existing CodeIgniter application as per documentation.
        
        Args:
            password: Plain text password
            
        Returns:
            MD5 hashed password
        """
        return hashlib.md5(password.encode('utf-8')).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Plain text password
            hashed_password: MD5 hashed password from database
            
        Returns:
            True if password matches
        """
        return UserModel.hash_password(password) == hashed_password
    
    @staticmethod
    def getUserByEmail(email: str, role: str = None) -> Optional[Dict[str, Any]]:
        """
        Get user by email with role information via JOIN
        
        Args:
            email: User email address
            role: Optional role name to filter by
            
        Returns:
            User dict with role information or None
        """
        try:
            if role:
                query = """
                    SELECT 
                        u.id,
                        u.first_name,
                        u.last_name,
                        u.email,
                        u.password,
                        u.is_active,
                        u.role_id,
                        u.password_reset_token,
                        u.created_at,
                        u.updated_at,
                        r.id as role_table_id,
                        r.role_name
                    FROM tr_users u
                    LEFT JOIN tr_roles r ON u.role_id = r.id
                    WHERE u.email = %s AND r.role_name = %s
                """
                result = db.execute_query(query, (email, role), fetch_one=True)
            else:
                query = """
                    SELECT 
                        u.id,
                        u.first_name,
                        u.last_name,
                        u.email,
                        u.password,
                        u.is_active,
                        u.role_id,
                        u.password_reset_token,
                        u.created_at,
                        u.updated_at,
                        r.id as role_table_id,
                        r.role_name
                    FROM tr_users u
                    LEFT JOIN tr_roles r ON u.role_id = r.id
                    WHERE u.email = %s
                """
                result = db.execute_query(query, (email,), fetch_one=True)
            
            return result
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {e}")
            return None
    
    @staticmethod
    def getUserById(user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user by ID with role information
        
        Args:
            user_id: User ID
            
        Returns:
            User dict with role information or None
        """
        try:
            query = """
                SELECT 
                    u.id,
                    u.first_name,
                    u.last_name,
                    u.email,
                    u.password,
                    u.is_active,
                    u.role_id,
                    u.password_reset_token,
                    u.created_at,
                    u.updated_at,
                    r.id as role_table_id,
                    r.role_name
                FROM tr_users u
                LEFT JOIN tr_roles r ON u.role_id = r.id
                WHERE u.id = %s
            """
            result = db.execute_query(query, (user_id,), fetch_one=True)
            return result
        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {e}")
            return None
    
    @staticmethod
    def getRoleId(role_name: str) -> Optional[int]:
        """
        Get role ID by role name
        
        Args:
            role_name: Role name (admin, employee, clients, applicant)
            
        Returns:
            Role ID or None
        """
        try:
            query = "SELECT id FROM tr_roles WHERE role_name = %s"
            result = db.execute_query(query, (role_name,), fetch_one=True)
            return result['id'] if result else None
        except Exception as e:
            logger.error(f"Error getting role ID for {role_name}: {e}")
            return None
    
    @staticmethod
    def register(user_info: Dict[str, Any], role: str) -> Optional[int]:
        """
        Register a new user (transaction-based)
        Inserts into tr_users and tr_user_details tables
        
        Args:
            user_info: Dictionary containing user information
            role: Role name to assign (ROLE_CLIENTS for signup)
            
        Returns:
            User ID if successful, None otherwise
        """
        try:
            # Get role ID
            role_id = UserModel.getRoleId(role)
            if not role_id:
                logger.error(f"Role {role} not found")
                return None
            
            # Hash password
            password = user_info.get('password', '')
            hashed_password = UserModel.hash_password(password)
            
            # Extract user data
            first_name = user_info.get('first_name', user_info.get('name', '').split()[0] if user_info.get('name') else '')
            last_name = user_info.get('last_name', ' '.join(user_info.get('name', '').split()[1:]) if user_info.get('name') and len(user_info.get('name', '').split()) > 1 else '')
            email = user_info.get('email', '')
            
            # Check if email already exists for this role
            existing_user = UserModel.getUserByEmail(email, role)
            if existing_user:
                logger.error(f"User with email {email} and role {role} already exists")
                return None
            
            # Use transaction context
            with db.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Insert into tr_users
                    user_insert_query = """
                        INSERT INTO tr_users 
                        (first_name, last_name, email, password, role_id, is_active, created_at)
                        VALUES (%s, %s, %s, %s, %s, 1, NOW())
                    """
                    cursor.execute(user_insert_query, (first_name, last_name, email, hashed_password, role_id))
                    user_id = cursor.lastrowid
                    
                    # Insert into tr_user_details
                    user_details_query = """
                        INSERT INTO tr_user_details 
                        (user_id, name, email, mobile, address_1, address_2, country_id, pincode, 
                         website, contact_name, contact_email, contact_phone, contact_address, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """
                    cursor.execute(
                        user_details_query,
                        (
                            user_id,
                            user_info.get('name', f"{first_name} {last_name}"),
                            email,
                            user_info.get('mobile', user_info.get('phone', '')),
                            user_info.get('address_1', user_info.get('address', '')),
                            user_info.get('address_2', ''),
                            user_info.get('country_id', None),
                            user_info.get('pincode', ''),
                            user_info.get('website', ''),
                            user_info.get('contact_name', ''),
                            user_info.get('contact_email', ''),
                            user_info.get('contact_phone', ''),
                            user_info.get('contact_address', '')
                        )
                    )
                    
                    conn.commit()
                    logger.info(f"User registered successfully: {email} with role {role}")
                    return user_id
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error registering user: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return None
    
    @staticmethod
    def updateUserPassword(user_id: int, password: str) -> bool:
        """
        Update user password
        
        Args:
            user_id: User ID
            password: New plain text password
            
        Returns:
            True if successful
        """
        try:
            hashed_password = UserModel.hash_password(password)
            query = "UPDATE tr_users SET password = %s, updated_at = NOW() WHERE id = %s"
            rows_affected = db.execute_query(query, (hashed_password, user_id))
            return rows_affected > 0
        except Exception as e:
            logger.error(f"Error updating password for user {user_id}: {e}")
            return False
    
    @staticmethod
    def generatePasswordResetHash(user_id: int) -> Optional[str]:
        """
        Generate password reset token
        
        Args:
            user_id: User ID
            
        Returns:
            Reset token or None
        """
        try:
            token = secrets.token_urlsafe(32)
            query = "UPDATE tr_users SET password_reset_token = %s, updated_at = NOW() WHERE id = %s"
            rows_affected = db.execute_query(query, (token, user_id))
            if rows_affected > 0:
                return token
            return None
        except Exception as e:
            logger.error(f"Error generating reset token for user {user_id}: {e}")
            return None
    
    @staticmethod
    def checkResetToken(token: str) -> Optional[Dict[str, Any]]:
        """
        Validate password reset token
        
        Args:
            token: Reset token
            
        Returns:
            User dict if token is valid, None otherwise
        """
        try:
            query = """
                SELECT 
                    u.id,
                    u.first_name,
                    u.last_name,
                    u.email,
                    u.password,
                    u.is_active,
                    u.role_id,
                    r.role_name
                FROM tr_users u
                LEFT JOIN tr_roles r ON u.role_id = r.id
                WHERE u.password_reset_token = %s
            """
            result = db.execute_query(query, (token,), fetch_one=True)
            return result
        except Exception as e:
            logger.error(f"Error checking reset token: {e}")
            return None
    
    @staticmethod
    def clearResetToken(user_id: int) -> bool:
        """
        Clear password reset token after use
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful
        """
        try:
            query = "UPDATE tr_users SET password_reset_token = NULL, updated_at = NOW() WHERE id = %s"
            rows_affected = db.execute_query(query, (user_id,))
            return rows_affected > 0
        except Exception as e:
            logger.error(f"Error clearing reset token for user {user_id}: {e}")
            return False

