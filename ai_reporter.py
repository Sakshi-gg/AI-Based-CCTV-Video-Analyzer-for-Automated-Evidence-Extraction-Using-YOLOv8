import os
import sqlite3
import json
import requests
from PySide6.QtCore import QThread, Signal

class AIReporterThread(QThread):
    # Signals to communicate back with your Main GUI Window safely
    summary_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, case_id, db_path="forensic_evidence.db"):
        super().__init__()
        self.case_id = case_id
        self.db_path = db_path

    def run(self):
        """Runs the network communication safely in the background."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.error_occurred.emit("GEMINI_API_KEY environment variable not configured.")
            return

        # 1. Gather tracking logs from SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT case_name, date_processed FROM cases WHERE case_id = ?", (self.case_id,))
        case_meta = cursor.fetchone()
        if not case_meta:
            conn.close()
            self.error_occurred.emit("Case data could not be recovered from database.")
            return
        
        case_name, date_run = case_meta
        
        cursor.execute("""
            SELECT entity_label, class_type, MIN(timestamp_in_video), MAX(timestamp_in_video), COUNT(*) 
            FROM tracking_logs 
            WHERE case_id = ?
            GROUP BY entity_label
        """, (self.case_id,))
        
        entities = cursor.fetchall()
        conn.close()

        if not entities:
            self.summary_ready.emit("No verified entities tracked within selected parameters.")
            return

        # 2. Build the data summary prompt payload
        data_summary_str = ""
        for label, cls_type, start_t, end_t, dynamic_count in entities:
            data_summary_str += f"- Tracked ID: {label} ({cls_type}) | Entry: {start_t} | Exit: {end_t} | Track Footprint: Observed across {dynamic_count} evaluated frames.\n"

        prompt = f"""
        You are an advanced digital forensics expert and criminal intelligence analyst assisting a law enforcement investigation. 
        Review the following automated object tracking analytics data dump extracted from a video source and generate a formal, objective, paragraph-style Incident Narrative Report.

        --- CASE DETAILS ---
        Target File: {case_name}
        Analysis Session Date: {date_run}

        --- EXTRACTED TRACKING TIMELINE METRICS ---
        {data_summary_str}

        --- REPORT INSTRUCTIONS ---
        - Write a formal, concise, paragraph-style narrative report summarizing the observed activities.
        - Do NOT use bullet points or markdown bolding inside the text summary body.
        - Reference specific Tracked entity IDs (e.g., Person_001, Car_002) and their precise video timeline entry/exit envelopes.
        - Maintain an objective, scientific, unbiased forensic investigator tone. Do not assume criminal guilt.
        """

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                response_json = response.json()
                ai_text = response_json['candidates'][0]['content']['parts'][0]['text']
                self.summary_ready.emit(ai_text)
            else:
                self.error_occurred.emit(f"Server Error Code {response.status_code}")
        except Exception as e:
            self.error_occurred.emit(str(e))
