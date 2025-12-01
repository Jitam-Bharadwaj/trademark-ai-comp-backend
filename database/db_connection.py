"""
Database connection module for MySQL
Handles database connection pooling and operations
"""
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages MySQL database connections"""
    
    _connection_pool = None
    
    def __init__(self):
        """Initialize database connection"""
        self.host = Config.DATABASE_HOSTNAME
        self.port = Config.DATABASE_PORT
        self.user = Config.DATABASE_USERNAME
        self.password = Config.DATABASE_PASSWORD
        self.database = Config.DATABASE_NAME
        self.charset = 'utf8mb4'
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        connection = None
        try:
            connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                autocommit=False
            )
            yield connection
        except Exception as e:
            if connection and connection.is_connected():
                connection.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False):
        """
        Execute a database query
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_one: Return single row if True
            fetch_all: Return all rows if True
            fetch_one takes precedence over fetch_all
            
        Returns:
            Query result based on fetch flags
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if fetch_one:
                    result = cursor.fetchone()
                elif fetch_all:
                    result = cursor.fetchall()
                else:
                    result = cursor.rowcount
                
                conn.commit()
                return result
            except Exception as e:
                conn.rollback()
                logger.error(f"Query execution error: {e}")
                raise
            finally:
                cursor.close()
    
    def execute_many(self, query: str, params_list: list):
        """
        Execute query with multiple parameter sets
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                conn.rollback()
                logger.error(f"Bulk execution error: {e}")
                raise
            finally:
                cursor.close()


# Global database instance
db = DatabaseConnection()
