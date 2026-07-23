import sys
import time
import os 
import csv 
import cv2
import numpy as np
from ultralytics import YOLO
from ai_reporter import AIReporterThread

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QSlider, QComboBox, QLineEdit, QScrollArea, 
    QCheckBox, QGridLayout, QGroupBox, QSpinBox, QProgressBar, QMessageBox, QFileDialog, QDialog, QTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont

from video_worker import VideoWorker
from utils.time_utils import hms_to_seconds, seconds_to_min_sec_string, seconds_to_hms
from utils.color_utils import is_color_match

# FIX 1: Single clean import from report_generator
from report_generator import generate_report

# PHASE 2: SQLite database imports
from database import initialize_db, create_case, log_evidence_frame


class AnalyzerWindow(QMainWindow):
    
    SIDEBAR_STRETCH      = 3         
    MAIN_CONTENT_STRETCH = 7    
    VIDEO_STRETCH        = 1           
    GALLERY_STRETCH      = 1.1       
    
    GALLERY_THUMBNAIL_WIDTH  = 250 
    GALLERY_THUMBNAIL_HEIGHT = 150 
    N_COLS = 3 

    COCO_CLASSES_MAPPING = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
    }
    ALL_INVESTIGATION_CLASSES = ['person', 'bicycle', 'car', 'truck', 'bus']
    
    COLOR_START_ANALYSIS = "#00B050"
    COLOR_STOP_ANALYSIS  = "#D32F2F"
    COLOR_RESET_FILTERS  = "#FFC107"
    COLOR_CLEAR_EVIDENCE = "#F4511E"
    COLOR_UPLOAD_FILE    = "#007BFF"
    COLOR_REPORT_SAVE    = "#673AB7"
    COLOR_TEXT_WHITE     = "white"
    COLOR_TEXT_BLACK     = "black"


    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚨 CCTV Video Analyzer")
        self.setGeometry(100, 100, 1500, 900) 
        self.apply_dark_style()
        
        self.video_worker       = None
        self.video_fps          = 30.0 
        self.total_frames       = 0 
        self.current_video_path = None 
        self.gallery_frame_counter  = 0 
        self.total_evidence_frames  = 0 
        self.last_analysis_rate     = "0.00 FPS"
        
        self.evidence_log  = [] 
        self.output_dir    = "video_analysis_output"
        self.video_metadata = {}

        # FIX 2: dedup tracker
        self.last_evidence_time_sec = -999.0

        # PHASE 2: database case tracker
        self.current_case_id = None
        self.ai_reporter     = None 
        initialize_db()
        
        self.setup_ui()

    def apply_dark_style(self):
        COLOR_BACKGROUND     = "#0A0A0A"       
        COLOR_SECONDARY      = "#1E1E1E"        
        COLOR_ACCENT         = "#FFFFFF"           
        COLOR_TEXT           = "#F0F2F6"             
        COLOR_BORDER         = "#FFFFFF"           
        COLOR_SIDEBAR        = "#2C2C2C" 
        COLOR_PROGRESS_GREEN = "#4CAF50"
        
        dark_stylesheet = f"""
        QMainWindow {{ background-color: {COLOR_BACKGROUND}; }}
        QWidget {{ color: {COLOR_TEXT}; background-color: {COLOR_SECONDARY}; }}
        QLabel {{ color: {COLOR_TEXT}; padding: 2px; }}
        QSpinBox, QLineEdit, QComboBox {{
            background-color: {COLOR_SIDEBAR};
            border: 1px solid {COLOR_BORDER};
            padding: 5px;
            border-radius: 3px;
            color: {COLOR_TEXT};
        }}
        QSlider::groove:horizontal {{ 
            border: 1px solid {COLOR_BORDER}; 
            height: 8px; 
            background: {COLOR_SECONDARY}; 
            margin: 2px 0; 
            border-radius: 4px; 
        }}
        QSlider::handle:horizontal {{ 
            background: {COLOR_ACCENT}; 
            border: 1px solid {COLOR_ACCENT}; 
            width: 18px; 
            margin: -5px 0; 
            border-radius: 9px; 
        }}
        QPushButton {{ 
            background-color: #555555; 
            color: white; 
            border-radius: 5px; 
            padding: 8px; 
            border: 1px solid white;
            font-weight: bold;
        }}
        QPushButton:hover {{ 
            background-color: #666666; 
            border: 1px solid #666666; 
        }}
        QGroupBox {{ 
            border: 1px solid white; 
            margin-top: 2ex; 
            border-radius: 5px; 
            padding-top: 10px; 
            font-weight: bold; 
        }}
        QGroupBox::title {{ 
            subcontrol-origin: margin; 
            subcontrol-position: top center; 
            padding: 0 3px; 
            background-color: {COLOR_SECONDARY}; 
            color: white; 
        }}
        QScrollArea {{ border: none; background-color: {COLOR_SECONDARY}; }}
        QCheckBox {{ padding: 5px; margin: 2px 0; }}
        QCheckBox::indicator {{
            width: 15px;
            height: 15px;
            border: 2px solid {COLOR_BORDER};        
            border-radius: 3px;             
            background: #1E1E1E;  
        }}
        QCheckBox::indicator:checked {{ background: {COLOR_PROGRESS_GREEN}; }}
        QProgressBar {{
            border: 2px solid #FFFFFF;
            border-radius: 5px;
            background-color: #2C2C2C;
            color: white;
            text-align: center;
            height: 25px;
            font-weight: bold;
        }}
        QProgressBar::chunk {{
            background-color: {COLOR_PROGRESS_GREEN}; 
            border-radius: 3px;
        }}
        """
        self.setStyleSheet(dark_stylesheet)

    def setup_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.sidebar_scroll_area = QScrollArea()
        self.sidebar_scroll_area.setWidgetResizable(True)
        self.control_panel = QWidget()
        self.control_panel.setMinimumWidth(250) 
        self.control_panel.setStyleSheet("background-color: #2C2C2C;") 
        self.setup_sidebar()
        self.sidebar_scroll_area.setWidget(self.control_panel)
        main_layout.addWidget(self.sidebar_scroll_area, self.SIDEBAR_STRETCH) 

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_area.setStyleSheet("background-color: #0A0A0A;") 
        main_layout.addWidget(self.content_area, self.MAIN_CONTENT_STRETCH) 
        self.setup_content_area()

    def setup_sidebar(self):
        sidebar_layout = QVBoxLayout(self.control_panel)
        sidebar_layout.setContentsMargins(10, 10, 10, 10) 
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- File Input & Control ---
        upload_group = QGroupBox("File Input & Control")
        upload_layout = QVBoxLayout(upload_group)
        self.file_path_label = QLabel("Ready to select file...")
        
        self.upload_btn = QPushButton("📂 Select Video")
        self.upload_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_UPLOAD_FILE}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")
        self.upload_btn.clicked.connect(self.select_video_file)
        upload_layout.addWidget(self.file_path_label)
        upload_layout.addWidget(self.upload_btn)

        self.analyze_btn = QPushButton("▶️ START")
        self.analyze_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_START_ANALYSIS}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")
        self.analyze_btn.clicked.connect(self.start_analysis_from_ui)
        self.analyze_btn.setEnabled(False) 
        upload_layout.addWidget(self.analyze_btn)
        
        self.reset_btn = QPushButton("↩️ Reset Filters")
        self.reset_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_RESET_FILTERS}; color: {self.COLOR_TEXT_BLACK}; border: none; }}")
        self.reset_btn.clicked.connect(self.reset_filters)
        upload_layout.addWidget(self.reset_btn)
        
        self.clear_evidence_btn = QPushButton("🗑️ Clear Evidence")
        self.clear_evidence_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_CLEAR_EVIDENCE}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")
        self.clear_evidence_btn.clicked.connect(self.clear_evidence_gallery)
        upload_layout.addWidget(self.clear_evidence_btn)
        sidebar_layout.addWidget(upload_group)
        
        # --- Evidence Output & Report ---
        output_group = QGroupBox("Evidence Output & Report")
        output_layout = QVBoxLayout(output_group)
        self.output_dir_input = QLineEdit(self.output_dir)
        output_layout.addWidget(QLabel("Output Folder Name:"))
        output_layout.addWidget(self.output_dir_input)

        self.generate_report_btn = QPushButton("📦 Save Report & Frames")
        self.generate_report_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_REPORT_SAVE}; color: {self.COLOR_TEXT_WHITE}; border: none; }}") 
        self.generate_report_btn.clicked.connect(self.generate_report)
        self.generate_report_btn.setEnabled(False) 
        output_layout.addWidget(self.generate_report_btn)

        self.ai_summary_btn = QPushButton("🤖 Generate AI Summary")
        self.ai_summary_btn.setStyleSheet("QPushButton { background-color: #00838F; color: white; border: none; }")
        self.ai_summary_btn.clicked.connect(self.generate_ai_summary)
        self.ai_summary_btn.setEnabled(False)
        output_layout.addWidget(self.ai_summary_btn)
        sidebar_layout.addWidget(output_group)

        # --- Target Objects ---
        target_group = QGroupBox("Target Objects")
        self.target_layout = QVBoxLayout(target_group)
        self.checkboxes = {}
        for cls_name in self.ALL_INVESTIGATION_CLASSES:
            cb = QCheckBox(cls_name.capitalize())
            cb.setChecked(cls_name in ['person']) 
            self.checkboxes[cls_name] = cb
            self.target_layout.addWidget(cb)
        sidebar_layout.addWidget(target_group)

        # --- Speed Optimization ---
        speed_group = QGroupBox("Speed Optimization (For long videos)")
        speed_layout = QVBoxLayout(speed_group)
        speed_layout.addWidget(QLabel("Frame Skip Factor (N - 1 to 100):"))
        self.skip_spinbox = QSpinBox()
        self.skip_spinbox.setRange(1, 100) 
        self.skip_spinbox.setValue(5) 
        speed_layout.addWidget(self.skip_spinbox)
        speed_layout.addWidget(QLabel("Detection Sensitivity (Confidence):"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 90) 
        self.conf_slider.setValue(25) 
        self.conf_slider_label = QLabel(f"Current: {self.conf_slider.value() / 100:.2f}")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_slider_label.setText(f"Current: {v / 100:.2f}"))
        speed_layout.addWidget(self.conf_slider)
        speed_layout.addWidget(self.conf_slider_label)
        sidebar_layout.addWidget(speed_group)

        # --- Time Stamp Clip Filter ---
        time_group = QGroupBox("Time Stamp Clip Filter (HH:MM:SS)")
        time_layout = QGridLayout(time_group)
        time_layout.addWidget(QLabel("Start Time:"), 0, 0)
        self.start_time_input = QLineEdit("00:00:00") 
        time_layout.addWidget(self.start_time_input, 0, 1)
        time_layout.addWidget(QLabel("End Time:"), 1, 0)
        self.end_time_input = QLineEdit("99:99:99") 
        time_layout.addWidget(self.end_time_input, 1, 1)
        sidebar_layout.addWidget(time_group)

        # FIX 2: Deduplication Filter
        dedup_group = QGroupBox("Deduplication Filter")
        dedup_layout = QVBoxLayout(dedup_group)
        dedup_layout.addWidget(QLabel("Min. Gap Between Evidence (seconds):"))
        self.dedup_spinbox = QSpinBox()
        self.dedup_spinbox.setRange(0, 60)
        self.dedup_spinbox.setValue(2)
        self.dedup_spinbox.setToolTip("Set to 0 to disable. Prevents saving near-identical frames.")
        dedup_layout.addWidget(self.dedup_spinbox)
        sidebar_layout.addWidget(dedup_group)

        # --- Color Based Filters ---
        color_group = QGroupBox("Color Based Filters")
        color_layout = QVBoxLayout(color_group)
        self.color_select = QComboBox()
        self.color_select.addItems(["None", "Red", "Blue", "Green", "White", "Black", "Yellow"])
        color_layout.addWidget(QLabel("Primary Color to Track:"))
        color_layout.addWidget(self.color_select)
        sidebar_layout.addWidget(color_group)

    def setup_content_area(self):
        # --- Metrics Bar ---
        metrics_layout = QHBoxLayout()
        
        def create_metric_label(text):
            label = QLabel(text)
            label.setStyleSheet("""
                QLabel { 
                    background-color: #2C2C2C; 
                    border: 1px solid #FFFFFF; 
                    border-radius: 5px; 
                    padding: 5px 10px; 
                    font-weight: bold;
                }
            """)
            return label

        self.res_label             = create_metric_label("Resolution: N/A")
        self.fps_original_label    = create_metric_label("FPS (Original): N/A")
        self.total_frames_label    = create_metric_label("Total Frames: N/A")
        self.current_fps_metric    = create_metric_label("Analysis Rate: 0.00 FPS")
        self.total_evidence_metric = create_metric_label("Total Evidence Frames: 0")
        
        metrics_layout.addWidget(self.res_label)
        metrics_layout.addWidget(self.fps_original_label)
        metrics_layout.addWidget(self.total_frames_label)
        metrics_layout.addStretch() 
        metrics_layout.addWidget(self.current_fps_metric)
        metrics_layout.addWidget(self.total_evidence_metric)
        self.content_layout.addLayout(metrics_layout)

        # --- Live Feed ---
        self.video_label = QLabel("Upload a video to see the live analysis feed.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(200) 
        self.video_label.setMaximumHeight(600) 
        self.video_label.setScaledContents(False) 
        self.video_label.setStyleSheet("border: 2px dashed #FFFFFF; font-size: 18px; font-weight: bold; background-color: black;")
        self.content_layout.addWidget(self.video_label, self.VIDEO_STRETCH) 
        
        # --- Progress Bar ---
        progress_info_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready for Analysis")
        progress_info_layout.addWidget(self.progress_bar)
        
        self.processing_time_label = QLabel("")
        self.processing_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processing_time_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50; padding: 5px;")
        self.processing_time_label.hide() 
        progress_info_layout.addWidget(self.processing_time_label)
        self.content_layout.addLayout(progress_info_layout)
        self.content_layout.addWidget(QLabel("---"))

        # --- Evidence Gallery ---
        gallery_title = QLabel("🖼️ EVIDENCE GALLERY: DETECTED CLIPS (Live)")
        gallery_title.setFont(QFont("Sans Serif", 14, QFont.Weight.Bold))
        gallery_title.setStyleSheet("color: #FFFFFF;")
        self.content_layout.addWidget(gallery_title)
        
        self.evidence_area = QScrollArea()
        self.evidence_area.setWidgetResizable(True)
        self.content_layout.addWidget(self.evidence_area, self.GALLERY_STRETCH) 
        
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        for i in range(self.N_COLS):
            self.gallery_layout.setColumnStretch(i, 1)
        self.gallery_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.evidence_area.setWidget(self.gallery_widget)
        
        self.no_evidence_label = QLabel("No target objects have been detected yet...")
        self.gallery_layout.addWidget(self.no_evidence_label, 0, 0)
        
    def extract_and_display_metadata(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.video_label.setText("Error: Could not open video file.")
            self.video_metadata = {}
            return

        video_width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps     = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        
        self.video_metadata = {
            "File Path":                video_path,
            "Filename":                 os.path.basename(video_path),
            "Resolution (W x H)":       f"{video_width} x {video_height}",
            "Frame Rate (FPS)":         f"{self.video_fps:.2f}",
            "Total Frames":             self.total_frames,
            "Duration":                 seconds_to_hms(total_duration_sec),
            "Codec (FourCC)":           "{:x}".format(int(cap.get(cv2.CAP_PROP_FOURCC))),
            "Video Format (Container)": os.path.splitext(video_path)[1].upper(),
            "Processing Date":          time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        cap.release()
        
        self.res_label.setText(f"Resolution: {video_width}x{video_height}")
        self.fps_original_label.setText(f"FPS (Original): {self.video_fps:.2f}")
        self.total_frames_label.setText(f"Total Frames: {self.total_frames}")

    @Slot()
    def select_video_file(self):
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker = None
            self.analyze_btn.setText("▶️ START")
            self.analyze_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_START_ANALYSIS}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")

        self.clear_evidence_gallery(reset_file_path=False) 
        self.progress_bar.setValue(0) 
        self.progress_bar.setFormat("Ready for Analysis")
        self.processing_time_label.hide() 
        self.current_fps_metric.setText("Analysis Rate: 0.00 FPS")
        self.total_frames   = 0
        self.video_metadata = {}
        
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.mov *.avi)"
        )
        
        if file_name:
            self.current_video_path = file_name 
            display_name = file_name.split('/')[-1]
            self.file_path_label.setText(f"File: {display_name}")
            self.analyze_btn.setEnabled(True) 
            self.extract_and_display_metadata(file_name) 
            
            if self.total_frames > 0 and self.video_fps > 0:
                total_duration_sec = self.total_frames / self.video_fps
                self.end_time_input.setText(seconds_to_hms(total_duration_sec))
            else:
                self.end_time_input.setText("99:99:99")
            
            self.start_time_input.setText("00:00:00")
            self.video_label.setText("Video loaded. Select filters and click 'START'.")
        else:
            self.file_path_label.setText("No file selected.")
            self.analyze_btn.setEnabled(False)
            self.current_video_path = None
            self.res_label.setText("Resolution: N/A")
            self.fps_original_label.setText("FPS (Original): N/A")
            self.total_frames_label.setText("Total Frames: N/A")

    @Slot()
    def start_analysis_from_ui(self):
        if not self.current_video_path:
            self.video_label.setText("Please select a video file first.")
            return

        start_sec    = hms_to_seconds(self.start_time_input.text())
        end_time_str = self.end_time_input.text()
        
        if end_time_str == "99:99:99":
            end_sec = float('inf')
        else:
            end_sec = hms_to_seconds(end_time_str)
        
        if start_sec == -1 or (end_sec != float('inf') and end_sec == -1) or (end_sec != float('inf') and start_sec >= end_sec):
            QMessageBox.warning(self, "Invalid Time Input", 
                                "Please ensure Start and End times are in HH:MM:SS format and Start < End.")
            return

        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker = None
            self.analyze_btn.setText("▶️ START")
            self.analyze_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_START_ANALYSIS}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")
            return

        color_filter = self.color_select.currentText()
        self.start_analysis(self.current_video_path, start_sec, end_sec, color_filter) 
        self.analyze_btn.setText("◼️ STOP")
        self.analyze_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_STOP_ANALYSIS}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")

    def get_current_filter_settings(self):
        target_classes_names = [name for name, cb in self.checkboxes.items() if cb.isChecked()]
        return {
            'target_objects': ", ".join(target_classes_names),
            'conf_threshold': f"{self.conf_slider.value() / 100.0:.2f}",
            'frame_skip':     self.skip_spinbox.value(),
            'start_time':     self.start_time_input.text(),
            'end_time':       self.end_time_input.text(),
            'color_filter':   self.color_select.currentText(),
        }

    def start_analysis(self, video_path, start_sec, end_sec, color_filter):
        self.processing_time_label.hide() 
        self.clear_evidence_gallery(reset_file_path=False)
        self.generate_report_btn.setEnabled(False)
        
        target_classes_names   = [name for name, cb in self.checkboxes.items() if cb.isChecked()]
        target_classes_indices = [k for k, v in self.COCO_CLASSES_MAPPING.items() if v in target_classes_names]
        
        if not target_classes_indices:
            self.video_label.setText("Please select target objects in the sidebar to begin.")
            self.analyze_btn.setText("▶️ START")
            self.analyze_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_START_ANALYSIS}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")
            return

        conf_threshold = self.conf_slider.value() / 100.0
        frame_skip     = self.skip_spinbox.value() 

        # PHASE 2: Create a new case in the database for this analysis run
        video_name           = os.path.basename(video_path)
        self.current_case_id = create_case(
            f"Case - {video_name} - {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.video_worker = VideoWorker(
            model_path='yolov8n.pt',
            target_classes=target_classes_indices, 
            conf_threshold=conf_threshold,
            frame_skip=frame_skip,
            video_path=video_path,
            video_fps=self.video_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            color_filter=color_filter,
            case_id=self.current_case_id
        )
        self.video_worker.frame_signal.connect(self.update_frame) 
        self.video_worker.finished_signal.connect(self.analysis_finished) 
        self.video_worker.start()

    @Slot()
    def reset_filters(self):
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker = None
            self.analyze_btn.setText("▶️ START")
            self.analyze_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_START_ANALYSIS}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")

        for cls_name, cb in self.checkboxes.items():
            cb.setChecked(cls_name == 'person')
            
        self.skip_spinbox.setValue(5) 
        self.conf_slider.setValue(25) 
        self.conf_slider_label.setText(f"Current: {self.conf_slider.value() / 100:.2f}")
        self.dedup_spinbox.setValue(2)  # FIX 2: reset dedup to default
        
        self.start_time_input.setText("00:00:00")
        if self.current_video_path and self.total_frames > 0 and self.video_fps > 0:
            total_duration_sec = self.total_frames / self.video_fps
            self.end_time_input.setText(seconds_to_hms(total_duration_sec))
        else:
            self.end_time_input.setText("99:99:99")
        
        self.color_select.setCurrentIndex(0) 
        file_part = self.current_video_path.split('/')[-1] if self.current_video_path else 'No file selected'
        self.file_path_label.setText(f"Filters reset. Ready to analyze: {file_part}")
        
    @Slot()
    def clear_evidence_gallery(self, reset_file_path=True):
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker = None
            self.analyze_btn.setText("▶️ START")
            self.analyze_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_START_ANALYSIS}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")

        for i in reversed(range(self.gallery_layout.count())): 
            widget_to_remove = self.gallery_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
                
        self.gallery_frame_counter  = 0 
        self.total_evidence_frames  = 0 
        self.total_evidence_metric.setText("Total Evidence Frames: 0")
        self.evidence_log           = []
        self.last_evidence_time_sec = -999.0  # FIX 2: reset dedup tracker
        self.generate_report_btn.setEnabled(False)
        self.gallery_layout.addWidget(self.no_evidence_label, 0, 0)
        
        if reset_file_path:
            self.video_label.setText("Evidence gallery cleared.")

    @Slot(np.ndarray, int, float, float, int)
    def update_frame(self, frame, detection_count, current_frame_pos, video_fps, frame_counter):
        current_time_real = time.time()
        
        if hasattr(self, 'prev_time_real'):
            time_diff = current_time_real - self.prev_time_real
            fps = 1.0 / time_diff if time_diff > 0 else 0
            self.last_analysis_rate = f"{fps:.2f} FPS" 
            self.current_fps_metric.setText(f"Analysis Rate: {self.last_analysis_rate}")
        self.prev_time_real = current_time_real
        
        if self.total_frames > 0:
            progress = int((current_frame_pos / self.total_frames) * 100)
            self.progress_bar.setValue(min(progress, 100)) 
            self.progress_bar.setFormat(f"Processing... {progress}% ({int(current_frame_pos)} / {self.total_frames} Frames)")
        else:
            self.progress_bar.setFormat(f"Processing... {int(current_frame_pos)} Frames")

        # --- Display Live Feed ---
        h, w, ch       = frame.shape
        frame_rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        q_img  = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))
        
        # --- Log Evidence ---
        if detection_count > 0:
            if self.no_evidence_label.parent():
                self.no_evidence_label.setParent(None) 
            filter_settings = self.get_current_filter_settings()
            self.add_evidence_to_gallery(
                frame_rgb.copy(),
                current_frame_pos, 
                video_fps, 
                detection_count,
                filter_settings
            )

    def add_evidence_to_gallery(self, frame_rgb, current_frame_pos, video_fps, detection_count, filter_settings):
        if current_frame_pos > 0:
            current_time_seconds = (current_frame_pos - 1) / video_fps
        else:
            current_time_seconds = 0.0

        # FIX 2: DEDUPLICATION CHECK
        min_gap = self.dedup_spinbox.value()
        if min_gap > 0 and (current_time_seconds - self.last_evidence_time_sec) < min_gap:
            return  # too soon since last saved frame, skip
        self.last_evidence_time_sec = current_time_seconds
        # END DEDUP CHECK

        timestamp_hms = seconds_to_hms(current_time_seconds)
            
        # Log to RAM list (for report generation)
        self.evidence_log.append({
            'frame_num':       int(current_frame_pos - 1),
            'timestamp':       timestamp_hms,
            'detection_count': detection_count,
            'frame_rgb':       frame_rgb,
            'filters_used':    filter_settings
        })

        # PHASE 2: Also persist to SQLite database
        if self.current_case_id:
            log_evidence_frame(
                case_id=self.current_case_id,
                video_filename=os.path.basename(self.current_video_path),
                frame_num=int(current_frame_pos - 1),
                timestamp=timestamp_hms,
                detection_count=detection_count,
                image_path="",  # filled in when report is saved
                filters_used=str(filter_settings)
            )
        
        # Build thumbnail widget
        container_widget = QWidget()
        container_layout = QVBoxLayout(container_widget)
        container_layout.setContentsMargins(5, 5, 5, 5) 
        container_layout.setSpacing(2) 
        
        image_label = QLabel()
        image_label.setFixedSize(self.GALLERY_THUMBNAIL_WIDTH, self.GALLERY_THUMBNAIL_HEIGHT) 
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        h, w, ch       = frame_rgb.shape
        bytes_per_line = ch * w
        q_img  = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        image_label.setPixmap(pixmap.scaled(
            image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))
        image_label.setStyleSheet("border: 2px solid #FFFFFF; background-color: black;") 
        
        caption_label = QLabel(f"Time: {timestamp_hms} | Detections: {detection_count}")
        caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        caption_label.setStyleSheet("font-size: 11px; font-weight: bold; padding: 2px;")

        container_layout.addWidget(image_label)
        container_layout.addWidget(caption_label)
        
        row = self.gallery_frame_counter // self.N_COLS
        col = self.gallery_frame_counter % self.N_COLS
        self.gallery_layout.addWidget(container_widget, row, col)
        
        self.gallery_frame_counter += 1
        self.total_evidence_frames += 1
        self.total_evidence_metric.setText(f"Total Evidence Frames: {self.total_evidence_frames}")
        self.evidence_area.verticalScrollBar().setValue(
            self.evidence_area.verticalScrollBar().maximum()
        )
        
    @Slot(float) 
    def analysis_finished(self, total_time):
        self.video_label.setText("Analysis Finished.")
        self.video_worker = None
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Analysis Complete (100%)")
        
        formatted_time_string = seconds_to_min_sec_string(total_time) 
        self.processing_time_label.setText(f"Total Processing Time: {formatted_time_string}")
        self.processing_time_label.show()
        self.current_fps_metric.setText(f"Analysis Rate: {self.last_analysis_rate}")

        if self.total_evidence_frames > 0:
            self.generate_report_btn.setEnabled(True)
            self.ai_summary_btn.setEnabled(True) 

        self.analyze_btn.setText("▶️ START")
        self.analyze_btn.setStyleSheet(f"QPushButton {{ background-color: {self.COLOR_START_ANALYSIS}; color: {self.COLOR_TEXT_WHITE}; border: none; }}")

    # ----------------------------------------------------------------
    # FIX 1: Delegates fully to report_generator.py
    # write_metadata_summary() removed — lives in report_generator now
    # ----------------------------------------------------------------
    @Slot()
    def generate_report(self):
        """Saves all logged evidence frames and generates report via report_generator module."""
        if not self.evidence_log:
            QMessageBox.information(self, "No Evidence", 
                                    "The evidence gallery is empty. Run an analysis first.")
            return

        output_dir_name = self.output_dir_input.text().strip() or "video_analysis_output"

        generate_report(
            evidence_log=self.evidence_log,
            output_dir_name=output_dir_name,
            video_metadata=self.video_metadata,
            last_analysis_rate=self.last_analysis_rate
        )

    @Slot()
    def generate_ai_summary(self):
        """Generates AI investigation summary using OpenRouter API."""
        if not self.current_case_id:
            QMessageBox.warning(self, "No Case", "Run an analysis first.")
            return

        self.ai_summary_btn.setEnabled(False)
        self.ai_summary_btn.setText("⏳ Generating...")

        self.ai_reporter = AIReporterThread(
            case_id=self.current_case_id,
            video_metadata=self.video_metadata
        )
        self.ai_reporter.summary_ready.connect(self.show_ai_summary)
        self.ai_reporter.error_occurred.connect(self.show_ai_error)
        self.ai_reporter.start()

    @Slot(str)
    def show_ai_summary(self, summary_text):
        """Displays the AI summary in a popup dialog."""
        self.ai_summary_btn.setEnabled(True)
        self.ai_summary_btn.setText("🤖 Generate AI Summary")
        
        if self.ai_reporter and self.ai_reporter.isRunning():
           self.ai_reporter.wait()
    
        dialog = QDialog(self)
        dialog.setWindowTitle("🤖 AI Investigation Summary")
        dialog.setMinimumSize(700, 500)
        dialog.setStyleSheet("background-color: #1E1E1E; color: #F0F2F6;")

        text_edit = QTextEdit()
        text_edit.setPlainText(summary_text)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #2C2C2C;
                color: #F0F2F6;
                font-size: 13px;
                padding: 15px;
                border: 1px solid #FFFFFF;
                border-radius: 5px;
            }
        """)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("background-color: #673AB7; color: white; padding: 8px; border: none;")
        close_btn.clicked.connect(dialog.accept)

        main_layout = QVBoxLayout(dialog)
        main_layout.addWidget(text_edit)
        main_layout.addWidget(close_btn)
        dialog.exec()

    @Slot(str)
    def show_ai_error(self, error_text):
        """Shows error if AI summary fails."""
        self.ai_summary_btn.setEnabled(True)
        self.ai_summary_btn.setText("🤖 Generate AI Summary")
        QMessageBox.critical(self, "AI Summary Error", error_text)

    def resizeEvent(self, event):
        if self.video_label.pixmap():
            self.video_label.setPixmap(self.video_label.pixmap().scaled(
                self.video_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
        super().resizeEvent(event)

    def closeEvent(self, event):
      if self.video_worker and self.video_worker.isRunning():
        self.video_worker.stop()
        self.video_worker = None
      if hasattr(self, 'ai_reporter') and self.ai_reporter.isRunning():
        self.ai_reporter.wait(3000)
        self.ai_reporter = None
      event.accept()
