"""
Verification script for authentication system setup
Run this to check if everything is configured correctly
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def check_imports():
    """Check if all required modules can be imported"""
    print("=" * 60)
    print("Checking Module Imports...")
    print("=" * 60)
    
    checks = []
    
    # Check database module
    try:
        import mysql.connector
        print("✓ Database connection module: OK")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Database connection module: FAILED - {e}")
        print("  → Install: pip install mysql-connector-python")
        checks.append(False)
    except Exception as e:
        print(f"⚠ Database connection module: Import OK but error - {e}")
        checks.append(True)  # Import works, connection might fail later
    
    # Check user model
    try:
        from models.user_model import UserModel, ROLE_ADMIN, ROLE_CLIENTS
        print("✓ User model module: OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ User model module: FAILED - {e}")
        checks.append(False)
    
    # Check session manager
    try:
        from auth.session_manager import initUserSession, initClientSession
        print("✓ Session manager module: OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ Session manager module: FAILED - {e}")
        checks.append(False)
    
    # Check session store
    try:
        from auth.session_store import create_session, get_session
        print("✓ Session store module: OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ Session store module: FAILED - {e}")
        checks.append(False)
    
    # Check dependencies
    try:
        from auth.dependencies import requireAdminAuth, requireClientAuth
        print("✓ Auth dependencies module: OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ Auth dependencies module: FAILED - {e}")
        checks.append(False)
    
    # Check auth routes
    try:
        from api.auth_routes import router
        print("✓ Auth routes module: OK")
        checks.append(True)
    except Exception as e:
        print(f"✗ Auth routes module: FAILED - {e}")
        checks.append(False)
    
    return all(checks)


def check_config():
    """Check configuration"""
    print("\n" + "=" * 60)
    print("Checking Configuration...")
    print("=" * 60)
    
    try:
        from config import Config
        
        # Database config
        print(f"Database Host: {Config.DATABASE_HOSTNAME}")
        print(f"Database Port: {Config.DATABASE_PORT}")
        print(f"Database Name: {Config.DATABASE_NAME}")
        print(f"Database User: {Config.DATABASE_USERNAME}")
        print(f"Database Password: {'*' * len(Config.DATABASE_PASSWORD) if Config.DATABASE_PASSWORD else 'NOT SET'}")
        
        if not Config.DATABASE_PASSWORD:
            print("⚠ Warning: DATABASE_PASSWORD not set")
        
        print(f"API Host: {Config.API_HOST}")
        print(f"API Port: {Config.API_PORT}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration check failed: {e}")
        return False


def check_database_connection():
    """Test database connection"""
    print("\n" + "=" * 60)
    print("Checking Database Connection...")
    print("=" * 60)
    
    try:
        from database.db_connection import db
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            # Test query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print("✓ Database connection: OK")
            
            # Check if tables exist
            cursor.execute("SHOW TABLES LIKE 'tr_%'")
            tables = cursor.fetchall()
            
            required_tables = ['tr_roles', 'tr_users', 'tr_user_details']
            # Handle both dict (pymysql) and tuple (mysql-connector) cursors
            if tables and isinstance(tables[0], dict):
                table_names = [list(table.values())[0] for table in tables]
            else:
                table_names = [table[0] for table in tables]
            
            print(f"\nFound {len(table_names)} tables:")
            for table in table_names:
                print(f"  - {table}")
            
            missing_tables = [t for t in required_tables if t not in table_names]
            if missing_tables:
                print(f"\n⚠ Warning: Missing required tables: {', '.join(missing_tables)}")
                print("  → Create these tables as per AUTHENTICATION_SYSTEM_DOCUMENTATION.md")
                return False
            
            print("\n✓ All required tables exist")
            return True
            
    except ImportError:
        print("✗ Database module not available")
        return False
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("  → Check database credentials in config.py or .env")
        print("  → Ensure MySQL server is running")
        return False


def check_fastapi_app():
    """Check if FastAPI app can be imported"""
    print("\n" + "=" * 60)
    print("Checking FastAPI Application...")
    print("=" * 60)
    
    try:
        from main import app
        print("✓ FastAPI app: OK")
        
        # Check if auth router is included
        routes = [route.path for route in app.routes]
        auth_routes = [r for r in routes if r.startswith('/auth')]
        
        if auth_routes:
            print(f"✓ Found {len(auth_routes)} authentication routes:")
            for route in auth_routes[:5]:  # Show first 5
                print(f"  - {route}")
            if len(auth_routes) > 5:
                print(f"  ... and {len(auth_routes) - 5} more")
        else:
            print("⚠ Warning: No authentication routes found")
            print("  → Auth routes might not be registered")
        
        return True
    except Exception as e:
        print(f"✗ FastAPI app check failed: {e}")
        return False


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("Authentication System Setup Verification")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run checks
    results.append(("Imports", check_imports()))
    results.append(("Configuration", check_config()))
    results.append(("Database Connection", check_database_connection()))
    results.append(("FastAPI App", check_fastapi_app()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready to use.")
        print("\nTo start the server, run:")
        print("  python scripts/run_api_server.py")
        print("\nOr:")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above before starting the server.")
        print("\nQuick fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check database configuration in config.py or .env")
        print("  - Ensure MySQL server is running")
        print("  - Verify database tables exist")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

