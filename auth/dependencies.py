"""
Authentication Dependencies for FastAPI
Implements route protection using dependencies
"""
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.user_model import ROLE_ADMIN, ROLE_CLIENTS
import logging

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def get_session_data(request: Request) -> Dict[str, Any]:
    """
    Get session data from request
    FastAPI doesn't have built-in sessions, so we'll use a session store
    In production, consider using redis or database-backed sessions
    
    Args:
        request: FastAPI request object
        
    Returns:
        Session data dictionary
    """
    # For now, we'll check cookies for session token
    # In a real implementation, you might use:
    # 1. JWT tokens in Authorization header
    # 2. Session cookies with backend session storage
    # 3. Redis sessions
    
    session_token = request.cookies.get('session_token')
    
    # This is a placeholder - in production, look up session in Redis/DB
    # For now, we'll implement a simple in-memory store
    # TODO: Replace with proper session storage
    from auth.session_store import get_session
    
    if session_token:
        return get_session(session_token)
    
    return {}


def requireAdminAuth(request: Request = Depends()) -> Dict[str, Any]:
    """
    Dependency for admin route protection
    Similar to AdminAuth filter in CodeIgniter
    
    Args:
        request: FastAPI request object
        
    Returns:
        User data dictionary
        
    Raises:
        HTTPException: If not authenticated or not admin role
    """
    session_data = get_session_data(request)
    
    if not session_data.get('is_logged_in'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please login.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user_data = session_data.get('user_data')
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session data"
        )
    
    if user_data.get('role') != ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Admin role required."
        )
    
    return user_data


def requireClientAuth(request: Request = Depends()) -> Dict[str, Any]:
    """
    Dependency for client route protection
    Similar to ClientAuth filter in CodeIgniter
    
    Args:
        request: FastAPI request object
        
    Returns:
        User data dictionary
        
    Raises:
        HTTPException: If not authenticated or not client role
    """
    session_data = get_session_data(request)
    
    if not session_data.get('is_client_logged_in'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please login.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    client_data = session_data.get('client_data')
    
    if not client_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session data"
        )
    
    if client_data.get('role') != ROLE_CLIENTS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Client role required."
        )
    
    return client_data


def requireAnyAuth(request: Request = Depends()) -> Dict[str, Any]:
    """
    Dependency for any authenticated user (admin or client)
    
    Args:
        request: FastAPI request object
        
    Returns:
        User data dictionary
        
    Raises:
        HTTPException: If not authenticated
    """
    session_data = get_session_data(request)
    
    # Check admin session
    if session_data.get('is_logged_in') and session_data.get('user_data'):
        return session_data['user_data']
    
    # Check client session
    if session_data.get('is_client_logged_in') and session_data.get('client_data'):
        return session_data['client_data']
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please login.",
        headers={"WWW-Authenticate": "Bearer"}
    )

