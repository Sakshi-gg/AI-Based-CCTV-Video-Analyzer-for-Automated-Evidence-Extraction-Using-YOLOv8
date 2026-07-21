"""
database.py
Persistent SQLite storage for CCTV Video Analyzer evidence.
Replaces the in-memory evidence_log list with a proper case management system.
"""

import sqlite3
import os
import time


DB_PATH = "cctv_evidence.db"


def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    """
    Creates all tables if they don't exist yet.
    Call this once at app startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evidence_frames (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id          INTEGER NOT NULL,
            video_filename   TEXT,
            frame_num        INTEGER,
            timestamp        TEXT,
            detection_count  INTEGER,
            image_path       TEXT,
            filters_used     TEXT,
            created_at       TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        )
    """)

    conn.commit()
    conn.close()


def create_case(name):
    """
    Creates a new investigation case and returns its case_id.
    Called automatically when analysis starts.
    """
    conn = get_connection()
    cursor = conn.cursor()
    created_at = time.strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(
        "INSERT INTO cases (name, created_at) VALUES (?, ?)",
        (name, created_at)
    )
    case_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return case_id


def log_evidence_frame(case_id, video_filename, frame_num,
                        timestamp, detection_count, image_path, filters_used):
    """
    Saves one evidence frame record to the database.
    Called from add_evidence_to_gallery() for every saved frame.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evidence_frames
            (case_id, video_filename, frame_num, timestamp,
             detection_count, image_path, filters_used, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        case_id,
        video_filename,
        frame_num,
        timestamp,
        detection_count,
        image_path,
        filters_used,
        time.strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()


def get_all_cases():
    """Returns all cases as a list of dicts. Used for the Cases panel."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases ORDER BY created_at DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_frames_for_case(case_id):
    """Returns all evidence frames for a given case_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM evidence_frames WHERE case_id=? ORDER BY frame_num ASC",
        (case_id,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def delete_case(case_id):
    """Deletes a case and all its evidence frames."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM evidence_frames WHERE case_id=?", (case_id,))
    cursor.execute("DELETE FROM cases WHERE case_id=?", (case_id,))
    conn.commit()
    conn.close()
