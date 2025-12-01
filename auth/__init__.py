"""
Authentication module initialization
"""
from .session_manager import (
    initUserSession,
    initClientSession,
    getAuthUser,
    clearAdminSession,
    clearClientSession
)

__all__ = [
    'initUserSession',
    'initClientSession',
    'getAuthUser',
    'clearAdminSession',
    'clearClientSession'
]

