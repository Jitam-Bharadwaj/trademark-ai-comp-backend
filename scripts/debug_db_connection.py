#!/usr/bin/env python3
"""
Debug script to test various pymysql connection parameters
"""
import pymysql
import ssl
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

def test_connection(name, **kwargs):
    print(f"\nTesting: {name}")
    print(f"Parameters: {kwargs}")
    
    try:
        # Merge with base config
        params = {
            'host': Config.DATABASE_HOSTNAME,
            'port': Config.DATABASE_PORT,
            'user': Config.DATABASE_USERNAME,
            'password': Config.DATABASE_PASSWORD,
            'database': Config.DATABASE_NAME,
            'charset': 'utf8mb4',
            **kwargs
        }
        
        conn = pymysql.connect(**params)
        print("✓ SUCCESS!")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

print("=" * 60)
print("Debugging Database Connection")
print("=" * 60)

# 1. Standard connection (Baseline)
test_connection("Standard Connection")

# 2. SSL Disabled
test_connection("SSL Disabled", ssl=None)

# 3. SSL Enabled (Default context)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
test_connection("SSL Enabled (No Verify)", ssl=ctx)

# 4. Auth Plugin: mysql_native_password
# Note: pymysql doesn't have a direct 'auth_plugin' param in connect(), 
# but we can try to see if it behaves differently. 
# Actually, pymysql supports auth plugins automatically.

# 5. Program Name (client_flag)
# Some servers might filter by client name? Unlikely but possible.

# 6. Compress
test_connection("Compression Enabled", compress=True)

# 7. Local Infile
test_connection("Local Infile Enabled", local_infile=True)

# 8. Defer Connect (Manual connect)
print("\nTesting: Defer Connect")
try:
    conn = pymysql.connect(
        host=Config.DATABASE_HOSTNAME,
        port=Config.DATABASE_PORT,
        user=Config.DATABASE_USERNAME,
        password=Config.DATABASE_PASSWORD,
        database=Config.DATABASE_NAME,
        defer_connect=True
    )
    conn.connect()
    print("✓ SUCCESS!")
    conn.close()
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 60)
