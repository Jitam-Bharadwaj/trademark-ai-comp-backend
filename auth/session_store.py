"""
Simple in-memory session store
In production, replace with Redis or database-backed sessions
"""
from typing import Dict, Any, Optional
import secrets
import time
from datetime import timedelta

# In-memory session store
# Format: {session_token: {data: {...}, expires_at: timestamp}}
_sessions: Dict[str, Dict[str, Any]] = {}

# Session expiration time (24 hours)
SESSION_EXPIRY_HOURS = 24


def create_session(session_data: Dict[str, Any]) -> str:
    """
    Create a new session
    
    Args:
        session_data: Session data dictionary
        
    Returns:
        Session token
    """
    session_token = secrets.token_urlsafe(32)
    expires_at = time.time() + (SESSION_EXPIRY_HOURS * 3600)
    
    _sessions[session_token] = {
        'data': session_data,
        'expires_at': expires_at
    }
    
    return session_token


def get_session(session_token: str) -> Dict[str, Any]:
    """
    Get session data by token
    
    Args:
        session_token: Session token
        
    Returns:
        Session data dictionary or empty dict
    """
    if not session_token:
        return {}
    
    session_info = _sessions.get(session_token)
    
    if not session_info:
        return {}
    
    # Check expiration
    if time.time() > session_info['expires_at']:
        # Session expired, remove it
        del _sessions[session_token]
        return {}
    
    return session_info['data']


def update_session(session_token: str, session_data: Dict[str, Any]) -> bool:
    """
    Update session data
    
    Args:
        session_token: Session token
        session_data: New session data
        
    Returns:
        True if successful
    """
    if session_token not in _sessions:
        return False
    
    _sessions[session_token]['data'] = session_data
    return True


def delete_session(session_token: str) -> bool:
    """
    Delete session
    
    Args:
        session_token: Session token
        
    Returns:
        True if successful
    """
    if session_token in _sessions:
        del _sessions[session_token]
        return True
    return False


def cleanup_expired_sessions():
    """
    Remove expired sessions
    Should be called periodically in production
    """
    current_time = time.time()
    expired_tokens = [
        token for token, info in _sessions.items()
        if current_time > info['expires_at']
    ]
    
    for token in expired_tokens:
        del _sessions[token]
    
    return len(expired_tokens)


def get_session_count() -> int:
    """
    Get number of active sessions
    """
    return len(_sessions)

