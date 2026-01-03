import time
import cv2
import numpy as np
from ultralytics import YOLO
from PySide6.QtCore import QThread, Signal
from utils.color_utils import is_color_match

# frame_signal now emits frame, detection_count, current_frame_pos, video_fps, frame_counter
class VideoWorker(QThread):
    
    frame_signal = Signal(np.ndarray, int, float, float, int)  
    finished_signal = Signal(float) 
    
    # ADDED start_sec, end_sec, color_filter to the constructor
    def __init__(self, model_path, target_classes, conf_threshold, frame_skip, video_path, video_fps, start_sec, end_sec, color_filter, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.video_path = video_path
        self._is_running = True
        self.video_fps = video_fps
        self.start_sec = start_sec # New: Start time
        self.end_sec = end_sec     # New: End time
        self.color_filter = color_filter # New: Color filter
        
        self.model = YOLO(self.model_path)
        
    def run(self):
        start_time_real = time.time()
        
        cap = cv2.VideoCapture(self.video_path)
        frame_counter = 0 
        
        # Initial SEEK for Time Filtering
        if self.start_sec > 0:
            start_frame = int(self.start_sec * self.video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_counter = start_frame - 1

        while self._is_running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_counter += 1
            current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES) 
            
            # FIX: Calculate the accurate time of the frame *just read*.
            # current_frame_pos is the index of the *next* frame, so (current_frame_pos - 1) is the index of the current frame.
            time_of_current_frame_sec = (current_frame_pos - 1) / self.video_fps

            # --- TIME FILTERING LOGIC (Check End Time) ---
            # Use the calculated accurate time for the filter check
            if self.end_sec != float('inf') and time_of_current_frame_sec > self.end_sec:
                break
            
            # --- FRAME SKIPPING LOGIC ---
            if self.frame_skip > 1 and frame_counter % self.frame_skip != 0:
                continue

            # --- Perform Object Detection (YOLO Inference) on the ORIGINAL frame ---
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                classes=self.target_classes,
                verbose=False
            )

            detections = results[0].boxes.cpu().numpy()
            validated_detections = []
            
            # --- COLOR FILTERING LOGIC (Check detections) ---
            if self.color_filter.lower() != 'none':
                for box_data in detections:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                    
                    # Ensure coordinates are within frame bounds (CRITICAL for slicing)
                    h, w, _ = frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Crop the bounding box area (Region of Interest - ROI)
                    roi = frame[y1:y2, x1:x2]
                    
                    # Validate the detection based on color match within the ROI
                    if is_color_match(roi, self.color_filter):
                        validated_detections.append(box_data)
            else:
                # If no color filter, all YOLO detections are validated
                validated_detections = detections
                
            # --- Frame Annotation and Metrics ---
            current_detection_count = len(validated_detections)

            # Start with a copy of the original frame
            annotated_frame = frame.copy() 
            
            # Manually draw bounding boxes and labels ONLY for the validated detections
            if current_detection_count > 0:
                names = self.model.names 
                color = (0, 255, 0) # Green box for clarity

                for box_data in validated_detections:
                    x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                    conf = box_data.conf[0]
                    cls = int(box_data.cls[0])
                    label = f"{names[cls]} {conf:.2f}"
                    
                    # Drawing the box and text using OpenCV
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Emit the frame, detection count, and frame position
            self.frame_signal.emit(
                annotated_frame, 
                current_detection_count, 
                current_frame_pos, 
                self.video_fps,
                frame_counter
            )
            
        cap.release()
        
        end_time_real = time.time()
        total_time = end_time_real - start_time_real
        self.finished_signal.emit(total_time)

    def stop(self):
        self._is_running = False
        self.wait()

