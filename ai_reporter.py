import os
import sqlite3
import requests
from database import DB_PATH
from PySide6.QtCore import QThread, Signal
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class AIReporterThread(QThread):
    summary_ready  = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, case_id, video_metadata=None):
        super().__init__()
        self.case_id        = case_id
        self.video_metadata = video_metadata or {}

    def run(self):
        # Resolve API key
        api_key = OPENROUTER_API_KEY
        if not api_key:
            self.error_occurred.emit("OPENROUTER_API_KEY not found in .env file.")
            return

        # Pull case data from database
        try:
            conn    = sqlite3.connect(DB_PATH)
            cursor  = conn.cursor()

            cursor.execute(
                "SELECT name, created_at FROM cases WHERE case_id=?",
                (self.case_id,)
            )
            case_meta = cursor.fetchone()
            if not case_meta:
                conn.close()
                self.error_occurred.emit("Case not found in database.")
                return

            case_name, date_run = case_meta

            cursor.execute("""
                SELECT entity_label, class_type,
                       MIN(video_time) as first_seen,
                       MAX(video_time) as last_seen,
                       COUNT(*)        as appearances
                FROM entity_tracks
                WHERE case_id=?
                GROUP BY entity_label
                ORDER BY appearances DESC
            """, (self.case_id,))
            entities = cursor.fetchall()

            cursor.execute(
                "SELECT COUNT(*) FROM evidence_frames WHERE case_id=?",
                (self.case_id,)
            )
            evidence_count = cursor.fetchone()[0]
            conn.close()

        except Exception as e:
            self.error_occurred.emit(f"Database error: {str(e)}")
            return

        if not entities:
            self.summary_ready.emit("No tracked entities found for this case.")
            return

        # Build entity summary string for prompt
        entity_lines = ""
        for label, cls_type, first_seen, last_seen, count in entities:
            entity_lines += (
                f"- {label} ({cls_type}): first seen at {first_seen}, "
                f"last seen at {last_seen}, "
                f"observed across {count} frames\n"
            )

        # Video metadata string
        video_info = ""
        if self.video_metadata:
            video_info = (
                f"Video File: {self.video_metadata.get('Filename', 'Unknown')}\n"
                f"Resolution: {self.video_metadata.get('Resolution (W x H)', 'Unknown')}\n"
                f"Duration: {self.video_metadata.get('Duration', 'Unknown')}\n"
                f"Frame Rate: {self.video_metadata.get('Frame Rate (FPS)', 'Unknown')} FPS\n"
                f"Evidence Frames Saved: {evidence_count}\n"
            )

        prompt = f"""
You are an advanced digital forensics expert and criminal intelligence analyst 
assisting a law enforcement investigation.

Review the following automated object tracking analytics extracted from CCTV footage 
and generate a formal, objective, paragraph-style Incident Narrative Report.

--- CASE DETAILS ---
Case: {case_name}
Analysis Date: {date_run}
{video_info}

--- TRACKED ENTITY TIMELINE ---
{entity_lines}

--- REPORT INSTRUCTIONS ---
- Write a formal 3-4 paragraph narrative report summarizing observed activities.
- Do NOT use bullet points or markdown inside the report body.
- Reference specific entity IDs (Person_001, Car_002) and their timestamps precisely.
- Maintain an objective, scientific, unbiased forensic investigator tone.
- Do not assume criminal guilt.
- End with recommended next steps for the investigator.
- Keep under 300 words.
"""

        # Call OpenRouter API (free, Python 3.8 compatible, no library needed)
        try:
            url     = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json"
            }
            payload = {
		 "model": "nvidia/nemotron-3-nano-30b-a3b:free",
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                ai_text = response.json()['choices'][0]['message']['content']
                self.summary_ready.emit(ai_text)
            else:
                self.error_occurred.emit(
                    f"API Error {response.status_code}: {response.text}"
                )
        except Exception as e:
            self.error_occurred.emit(f"OpenRouter API error: {str(e)}")
