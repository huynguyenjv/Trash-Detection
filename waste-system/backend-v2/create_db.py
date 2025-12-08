"""
Database Initialization Script
Creates PostgreSQL database if it doesn't exist
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys


def create_database():
    """Create waste_detection database if not exists"""
    
    # Connection parameters
    host = "localhost"
    port = 5432
    user = "postgres"
    password = "baohuy2501"
    db_name = "waste_detection"
    
    print(f"🔌 Connecting to PostgreSQL server at {host}:{port}...")
    
    try:
        # Connect to PostgreSQL server (default postgres database)
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        exists = cursor.fetchone()
        
        if exists:
            print(f"✅ Database '{db_name}' already exists")
        else:
            # Create database
            cursor.execute(f'CREATE DATABASE {db_name}')
            print(f"✅ Database '{db_name}' created successfully")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure PostgreSQL is running")
        print("   2. Check username and password")
        print("   3. Verify PostgreSQL is listening on localhost:5432")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = create_database()
    sys.exit(0 if success else 1)
