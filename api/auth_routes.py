"""
Authentication Routes
Implements signup, login, and logout endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from typing import Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.user_model import UserModel, ROLE_ADMIN, ROLE_CLIENTS
from auth.session_manager import initUserSession, initClientSession, clearAdminSession, clearClientSession
from auth.session_store import create_session, delete_session
from auth.dependencies import get_session_data
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request model"""
    email: EmailStr
    password: str = Field(..., min_length=6)


class ClientSignupRequest(BaseModel):
    """Client signup request model"""
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    email: EmailStr
    password: str = Field(..., min_length=6)
    password_confirmation: str = Field(..., alias="password_confirmation")
    mobile: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    address_1: Optional[str] = None
    country_id: Optional[int] = None
    pincode: Optional[str] = None
    website: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None
    contact_address: Optional[str] = None
    
    @model_validator(mode='after')
    def passwords_match(self):
        if self.password != self.password_confirmation:
            raise ValueError('Passwords do not match')
        return self
    
    class Config:
        populate_by_name = True


class LoginResponse(BaseModel):
    """Login response model"""
    success: bool
    message: str
    user_data: Optional[dict] = None
    session_token: Optional[str] = None


class SignupResponse(BaseModel):
    """Signup response model"""
    success: bool
    message: str
    user_id: Optional[int] = None


class LogoutResponse(BaseModel):
    """Logout response model"""
    success: bool
    message: str


@router.post("/admin/login", response_model=LoginResponse, tags=["Authentication"])
async def admin_login(request: LoginRequest, response: Response):
    """
    Admin login endpoint
    
    Args:
        request: Login credentials
        response: FastAPI response object
        
    Returns:
        Login response with session token
    """
    try:
        # Get user by email and role
        user = UserModel.getUserByEmail(request.email, ROLE_ADMIN)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Verify password (MD5 hash comparison as per documentation)
        if not UserModel.verify_password(request.password, user['password']):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password"
            )
        
        # Check if user is active
        if not user.get('is_active', 1):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Initialize session
        session_data = initUserSession(request.email, ROLE_ADMIN)
        
        # Create session token
        session_token = create_session(session_data)
        
        # Set session cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            max_age=86400,  # 24 hours
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        
        logger.info(f"Admin login successful: {request.email}")
        
        return LoginResponse(
            success=True,
            message="Login successful",
            user_data=session_data['user_data'],
            session_token=session_token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/client/login", response_model=LoginResponse, tags=["Authentication"])
async def client_login(request: LoginRequest, response: Response):
    """
    Client login endpoint
    
    Args:
        request: Login credentials
        response: FastAPI response object
        
    Returns:
        Login response with session token
    """
    try:
        # Get user by email and role
        user = UserModel.getUserByEmail(request.email, ROLE_CLIENTS)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Verify password (MD5 hash comparison as per documentation)
        if not UserModel.verify_password(request.password, user['password']):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password"
            )
        
        # Check if user is active
        if not user.get('is_active', 1):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Initialize session
        session_data = initClientSession(request.email, ROLE_CLIENTS)
        
        # Create session token
        session_token = create_session(session_data)
        
        # Set session cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            max_age=86400,  # 24 hours
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        
        logger.info(f"Client login successful: {request.email}")
        
        return LoginResponse(
            success=True,
            message="Login successful",
            user_data=session_data['client_data'],
            session_token=session_token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Client login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/client/signup", response_model=SignupResponse, tags=["Authentication"])
async def client_signup(request: ClientSignupRequest):
    """
    Client signup endpoint
    
    Args:
        request: Signup information
        
    Returns:
        Signup response with user ID
    """
    try:
        # Prepare user info dictionary
        user_info = {
            'first_name': request.first_name,
            'last_name': request.last_name,
            'email': request.email,
            'password': request.password,
            'name': f"{request.first_name} {request.last_name}",
            'mobile': request.mobile or request.phone,
            'phone': request.phone or request.mobile,
            'address': request.address or request.address_1,
            'address_1': request.address_1 or request.address,
            'country_id': request.country_id,
            'pincode': request.pincode,
            'website': request.website,
            'contact_name': request.contact_name,
            'contact_email': str(request.contact_email) if request.contact_email else None,
            'contact_phone': request.contact_phone,
            'contact_address': request.contact_address
        }
        
        # Register user
        user_id = UserModel.register(user_info, ROLE_CLIENTS)
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registration failed. Email may already exist."
            )
        
        logger.info(f"Client signup successful: {request.email}")
        
        return SignupResponse(
            success=True,
            message="Registration successful",
            user_id=user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Client signup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/admin/logout", response_model=LogoutResponse, tags=["Authentication"])
async def admin_logout(request: Request, response: Response):
    """
    Admin logout endpoint
    
    Args:
        request: FastAPI request object
        response: FastAPI response object
        
    Returns:
        Logout response
    """
    session_token = request.cookies.get('session_token')
    
    if session_token:
        delete_session(session_token)
    
    # Clear cookie
    response.delete_cookie(key="session_token")
    
    return LogoutResponse(
        success=True,
        message="Logout successful"
    )


@router.post("/client/logout", response_model=LogoutResponse, tags=["Authentication"])
async def client_logout(request: Request, response: Response):
    """
    Client logout endpoint
    
    Args:
        request: FastAPI request object
        response: FastAPI response object
        
    Returns:
        Logout response
    """
    session_token = request.cookies.get('session_token')
    
    if session_token:
        delete_session(session_token)
    
    # Clear cookie
    response.delete_cookie(key="session_token")
    
    return LogoutResponse(
        success=True,
        message="Logout successful"
    )


@router.get("/me", tags=["Authentication"])
async def get_current_user(session_data: dict = Depends(get_session_data)):
    """
    Get current authenticated user
    
    Args:
        session_data: Session data from dependency
        
    Returns:
        User information
    """
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    # Check admin session
    if session_data.get('is_logged_in') and session_data.get('user_data'):
        return {
            "authenticated": True,
            "user": session_data['user_data'],
            "type": "admin"
        }
    
    # Check client session
    if session_data.get('is_client_logged_in') and session_data.get('client_data'):
        return {
            "authenticated": True,
            "user": session_data['client_data'],
            "type": "client"
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated"
    )

