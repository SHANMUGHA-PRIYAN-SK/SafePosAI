# database.py
import sqlite3
import traceback
from datetime import datetime
# Import default DB name from config, but allow overriding
from config import DATABASE_NAME as DEFAULT_DB_NAME

def init_database(db_name=DEFAULT_DB_NAME):
    """Initializes the SQLite database and tables if they don't exist."""
    conn = None
    try:
        print(f"Initializing database: {db_name}")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Person Sessions Table (One entry per video processed)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS person_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_timestamp TEXT NOT NULL,
            video_path TEXT
        )
        ''')

        # Posture Data Table (Frame-by-frame data)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS posture_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            person_tracker_id INTEGER NOT NULL, -- Tracker ID assigned during the session
            frame_number INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            risk_score INTEGER,
            status TEXT,
            back_angle REAL,
            neck_angle REAL,
            arm_angle REAL,
            FOREIGN KEY (session_id) REFERENCES person_sessions(id) ON DELETE CASCADE
        )
        ''')
        # Add index for faster querying by session and person
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_posture_data_session_person
        ON posture_data (session_id, person_tracker_id);
        ''')


        # Stress Summary Table (One entry per person per session)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stress_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            person_tracker_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL, -- Timestamp when summary was generated
            total_frames INTEGER,
            safe_percent REAL,
            neutral_percent REAL,
            strain_percent REAL,
            avg_risk_score REAL,
            avg_back_angle REAL,
            avg_neck_angle REAL,
            avg_arm_angle REAL,
            FOREIGN KEY (session_id) REFERENCES person_sessions(id) ON DELETE CASCADE,
            UNIQUE (session_id, person_tracker_id) -- Ensure only one summary per person/session
        )
        ''')
        # Add index for faster querying by session and person
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stress_summary_session_person
        ON stress_summary (session_id, person_tracker_id);
        ''')


        conn.commit()
        print(f"Database '{db_name}' initialized successfully.")
        return conn
    except sqlite3.Error as e:
        print(f"Database error during initialization: {e}")
        print(traceback.format_exc())
        if conn:
            conn.rollback() # Rollback any partial changes
        return None
    except Exception as e:
        print(f"Unexpected error initializing database: {e}")
        print(traceback.format_exc())
        return None
    finally:
        # Don't close the connection here, return it for use
        pass


def get_tracking_data(person_id=None, session_id=None, db_name=DEFAULT_DB_NAME):
    """Queries the database for stress summary data."""
    conn = None
    results = []
    try:
        conn = sqlite3.connect(db_name)
        # Use row factory for dictionary results
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Base query selects all columns from stress_summary joined with session info
        query = """
            SELECT ss.*, ps.session_timestamp, ps.video_path
            FROM stress_summary ss
            JOIN person_sessions ps ON ss.session_id = ps.id
        """
        params = []
        conditions = []

        if person_id is not None:
            conditions.append("ss.person_tracker_id = ?")
            params.append(person_id)
        if session_id is not None:
            conditions.append("ss.session_id = ?")
            params.append(session_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Order by session time (most recent first), then by person ID
        query += " ORDER BY ps.session_timestamp DESC, ss.person_tracker_id"

        cursor.execute(query, params)
        # Fetch all results as dictionaries
        results = [dict(row) for row in cursor.fetchall()]

    except sqlite3.Error as e:
        print(f"Database query error: {e}")
        print(traceback.format_exc())
        # Return empty list on error
    except Exception as e:
        print(f"Unexpected error querying database: {e}")
        print(traceback.format_exc())
    finally:
        if conn:
            conn.close()

    return results # Return list of dictionaries