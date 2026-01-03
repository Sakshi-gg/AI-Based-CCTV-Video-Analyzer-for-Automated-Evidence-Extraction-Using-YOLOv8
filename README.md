# AI-Based CCTV Video Analyzer for Automated Evidence Extraction

An automated system leveraging **YOLOv8** for real-time CCTV footage analysis, designed to detect specific objects and extract video evidence efficiently.

---

## 🚀 Key Features
* **Intelligent Detection:** Utilizes YOLOv8 for high-accuracy object recognition.
* **Evidence Management:** Automatically extracts and saves relevant video segments.
* **User-Friendly GUI:** Built with PyQt5 for easy file selection and analysis tracking.
* **Automated Reporting:** Generates detailed `.odt` tables and summary reports.

---

## 📁 Project Structure
* `main.py` - Core application launcher.
* `gui.py` - Graphical User Interface logic.
* `video_worker.py` - Handles background video processing.
* `yolov8n.pt` - Pre-trained model weights.
* `utils/` - Utility scripts for processing and reporting.

---

## 🛠️ Quick Start

### 1. Requirements
Ensure you have Python installed, then install the necessary libraries:
```bash
pip install ultralytics PyQt5


