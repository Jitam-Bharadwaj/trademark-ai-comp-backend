#!/usr/bin/env python3
"""
Side-by-side comparison of pymysql and mysql-connector-python
with the CORRECT password.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
import pymysql
import mysql.connector

def test_pymysql():
    print("\nTesting PyMySQL...")
    try:
        conn = pymysql.connect(
            host=Config.DATABASE_HOSTNAME,
            port=Config.DATABASE_PORT,
            user=Config.DATABASE_USERNAME,
            password=Config.DATABASE_PASSWORD,
            database=Config.DATABASE_NAME,
            charset='utf8mb4'
        )
        print("✓ PyMySQL: SUCCESS!")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ PyMySQL: FAILED - {e}")
        return False

def test_mysql_connector():
    print("\nTesting MySQL Connector Python...")
    try:
        conn = mysql.connector.connect(
            host=Config.DATABASE_HOSTNAME,
            port=Config.DATABASE_PORT,
            user=Config.DATABASE_USERNAME,
            password=Config.DATABASE_PASSWORD,
            database=Config.DATABASE_NAME
        )
        print("✓ MySQL Connector: SUCCESS!")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ MySQL Connector: FAILED - {e}")
        return False

print("=" * 60)
print("Library Comparison")
print(f"Password: {Config.DATABASE_PASSWORD}")
print(f"Length: {len(Config.DATABASE_PASSWORD)}")
print("=" * 60)

pymysql_result = test_pymysql()
mysql_connector_result = test_mysql_connector()

print("\n" + "=" * 60)
print("Summary")
print(f"PyMySQL: {'✓ PASS' if pymysql_result else '✗ FAIL'}")
print(f"MySQL Connector: {'✓ PASS' if mysql_connector_result else '✗ FAIL'}")
print("=" * 60)
