import sqlite3
from datetime import datetime

class ForensicDatabase:
    def __init__(self, db_path="forensic_evidence.db"):
        self.db_path = db_path
        self.create_tables()

    def create_tables(self):
        """Initializes the persistent SQLite schema for multi-video forensic cases."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table 1: Stores individual case sessions so investigators can look up older runs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cases (
                case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_name TEXT NOT NULL UNIQUE,
                video_source_path TEXT NOT NULL,
                date_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table 2: Upgraded to support vehicle identification metadata columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracking_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER,
                entity_label TEXT NOT NULL,
                class_type TEXT NOT NULL,
                confidence_score REAL,
                frame_number INTEGER,
                timestamp_in_video TEXT NOT NULL,
                license_plate TEXT DEFAULT 'N/A',
                FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
            )
        ''')

        conn.commit()
        conn.close()

    def register_case(self, case_name, video_path):
        """Registers a video run. Returns the case_id row index."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO cases (case_name, video_source_path) 
                VALUES (?, ?)
            ''', (case_name, video_path))
            conn.commit()
            case_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            # If case name already exists, fetch the existing ID
            cursor.execute('SELECT case_id FROM cases WHERE case_name = ?', (case_name,))
            case_id = cursor.fetchone()[0]
        finally:
            conn.close()
        return case_id

    def log_entity_frame(self, case_id, entity_label, class_type, confidence, frame_num, video_time, license_plate='N/A'):
        """Saves a single target tracking confirmation point along with vehicle metadata right to storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO tracking_logs (case_id, entity_label, class_type, confidence_score, frame_number, timestamp_in_video,     license_plate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (case_id, entity_label, class_type, confidence, frame_num, video_time, license_plate))
        conn.commit()
        conn.close()
