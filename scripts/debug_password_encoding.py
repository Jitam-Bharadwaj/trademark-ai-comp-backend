#!/usr/bin/env python3
"""
Debug script to test password encoding variations
"""
import pymysql
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

def test_password_encoding(encoding_name, password_bytes):
    print(f"\nTesting Password Encoding: {encoding_name}")
    print(f"Bytes: {password_bytes}")
    
    try:
        # We need to monkey-patch or manually handle the connection to force specific bytes
        # But pymysql takes a string and encodes it. 
        # So we can try to pass a string that encodes to the desired bytes in the default charset (utf8mb4)
        # OR we can try to change the connection charset.
        
        # Strategy: Try to connect with 'charset' set to different values
        
        params = {
            'host': Config.DATABASE_HOSTNAME,
            'port': Config.DATABASE_PORT,
            'user': Config.DATABASE_USERNAME,
            'password': Config.DATABASE_PASSWORD,
            'database': Config.DATABASE_NAME,
            'charset': encoding_name
        }
        
        print(f"Connecting with charset='{encoding_name}'...")
        conn = pymysql.connect(**params)
        print("✓ SUCCESS!")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

print("=" * 60)
print("Debugging Password Encoding")
print("=" * 60)
print(f"Original Password: {Config.DATABASE_PASSWORD}")
print(f"UTF-8 Bytes: {Config.DATABASE_PASSWORD.encode('utf-8')}")
print(f"Latin-1 Bytes: {Config.DATABASE_PASSWORD.encode('latin1')}")
print("=" * 60)

# 1. UTF-8 (Default)
test_password_encoding('utf8mb4', None)

# 2. Latin-1
test_password_encoding('latin1', None)

# 3. CP1252
test_password_encoding('cp1252', None)

# 4. Binary (if possible)
# pymysql expects string for password, so we can't easily pass bytes directly 
# without modifying the library or using a specific charset that maps 1:1.
# Latin1 is usually the 1:1 map.

print("\n" + "=" * 60)
