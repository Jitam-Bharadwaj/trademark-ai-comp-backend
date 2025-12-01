"""
Session Management for Authentication
Implements session initialization functions as per documentation
"""
from typing import Optional, Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.user_model import UserModel, ROLE_ADMIN, ROLE_CLIENTS
import logging

logger = logging.getLogger(__name__)


def initUserSession(email: str, role: str) -> Dict[str, Any]:
    """
    Initialize admin session (as per documentation)
    
    Sets session keys:
    - is_logged_in = true
    - user_data = {user_id, first_name, last_name, email, role, is_active}
    
    Args:
        email: User email
        role: User role (should be ROLE_ADMIN)
        
    Returns:
        Session data dictionary
    """
    user = UserModel.getUserByEmail(email, role)
    
    if user is None:
        raise Exception(f"No user exists with this email - {email}")
    
    user_data = {
        'is_logged_in': True,
        'user_data': {
            'user_id': user['id'],
            'first_name': user['first_name'],
            'last_name': user['last_name'],
            'email': user['email'],
            'role': user['role_name'],
            'is_active': bool(user['is_active'])
        }
    }
    
    return user_data


def initClientSession(email: str, role: str) -> Dict[str, Any]:
    """
    Initialize client session (as per documentation)
    
    Sets session keys:
    - is_client_logged_in = true
    - client_data = {user_id, first_name, last_name, email, role, is_active}
    
    Args:
        email: User email
        role: User role (should be ROLE_CLIENTS)
        
    Returns:
        Session data dictionary
    """
    client = UserModel.getUserByEmail(email, role)
    
    if client is None:
        raise Exception(f"No user exists with this email - {email}")
    
    client_data = {
        'is_client_logged_in': True,
        'client_data': {
            'user_id': client['id'],
            'first_name': client['first_name'],
            'last_name': client['last_name'],
            'email': client['email'],
            'role': client['role_name'],
            'is_active': bool(client['is_active'])
        }
    }
    
    return client_data


def getAuthUser(session_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get authenticated user from session data
    
    Args:
        session_data: Session dictionary
        
    Returns:
        User data dictionary or None
    """
    # Check for admin session
    if session_data.get('is_logged_in') and session_data.get('user_data'):
        return session_data['user_data']
    
    # Check for client session
    if session_data.get('is_client_logged_in') and session_data.get('client_data'):
        return session_data['client_data']
    
    return None


def clearAdminSession() -> Dict[str, Any]:
    """
    Clear admin session data
    
    Returns:
        Empty session dictionary
    """
    return {
        'is_logged_in': False,
        'user_data': None
    }


def clearClientSession() -> Dict[str, Any]:
    """
    Clear client session data
    
    Returns:
        Empty session dictionary
    """
    return {
        'is_client_logged_in': False,
        'client_data': None
    }

