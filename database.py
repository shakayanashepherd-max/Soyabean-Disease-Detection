
import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "leaflensai.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Create the scan-history table."""

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                filename TEXT,
                prediction TEXT,
                confidence REAL,
                scan_time TEXT
            )
        """)

        conn.commit()
        print("Scan history table created successfully.")

    except sqlite3.Error as error:
        conn.rollback()
        print("Database initialization error:", error)
        raise

    finally:
        conn.close()


def save_scan(username, filename, prediction, confidence):
    conn = get_connection()

    try:
        conn.execute("""
            INSERT INTO scans
            (username, filename, prediction, confidence, scan_time)
            VALUES (?, ?, ?, ?, ?)
        """, (
            username,
            filename,
            prediction,
            float(confidence),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        return True

    except sqlite3.Error as error:
        print("Could not save scan:", error)
        return False

    finally:
        conn.close()


def get_user_scans(username):
    conn = get_connection()

    try:
        cursor = conn.execute("""
            SELECT *
            FROM scans
            WHERE username = ?
            ORDER BY id DESC
        """, (username,))

        return [dict(row) for row in cursor.fetchall()]

    except sqlite3.Error as error:
        print("Could not retrieve scan history:", error)
        return []

    finally:
        conn.close()
