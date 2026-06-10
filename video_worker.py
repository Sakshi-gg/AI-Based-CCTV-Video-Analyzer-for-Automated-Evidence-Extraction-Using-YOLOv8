import time
import cv2
import numpy as np
from ultralytics import YOLO
from PySide6.QtCore import QThread, Signal
from utils.color_utils import is_color_match
from database_manager import ForensicDatabase

class VideoWorker(QThread):
    frame_signal = Signal(np.ndarray, int, float, float, int)  
    finished_signal = Signal(float) 
    
    def __init__(self, model_path, target_classes, conf_threshold, frame_skip, 
                 video_path, video_fps, start_sec, end_sec, color_filter, 
                 case_id=1, db_path="forensic_evidence.db", parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.video_path = video_path
        self._is_running = True
        self.video_fps = video_fps
        self.start_sec = start_sec 
        self.end_sec = end_sec     
        self.color_filter = color_filter 
        
        self.case_id = case_id
        self.db = ForensicDatabase(db_path)
        self.model = YOLO(self.model_path)
        
    def run(self):
        start_time_real = time.time()
        cap = cv2.VideoCapture(self.video_path)
        frame_counter = 0 
        
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
            time_of_current_frame_sec = (current_frame_pos - 1) / self.video_fps

            if self.end_sec != float('inf') and time_of_current_frame_sec > self.end_sec:
                break
            
            if self.frame_skip > 1 and frame_counter % self.frame_skip != 0:
                continue

            results = self.model.track(
                source=frame,
                conf=self.conf_threshold,
                classes=self.target_classes,
                persist=True,
                tracker="botsort.yaml",
                verbose=False
            )

            validated_detections = []
            
            if results[0].boxes is not None:
                detections = results[0].boxes
                boxes_xyxy = detections.xyxy.cpu().numpy()
                confidences = detections.conf.cpu().numpy()
                classes = detections.cls.cpu().numpy().astype(int)
                
                if detections.id is not None:
                    track_ids = detections.id.cpu().numpy().astype(int)
                else:
                    track_ids = [None] * len(boxes_xyxy)
                
                if self.color_filter.lower() != 'none':
                    for idx, box in enumerate(boxes_xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        roi = frame[y1:y2, x1:x2]
                        if is_color_match(roi, self.color_filter):
                            validated_detections.append((box, confidences[idx], classes[idx], track_ids[idx]))
                else:
                    for idx, box in enumerate(boxes_xyxy):
                        validated_detections.append((box, confidences[idx], classes[idx], track_ids[idx]))
                    
            current_detection_count = len(validated_detections)
            annotated_frame = frame.copy() 
                
            if current_detection_count > 0:
                names = self.model.names 
                color = (0, 0, 255) 

                for box, conf, cls, t_id in validated_detections:
                    x1, y1, x2, y2 = map(int, box)
                    class_name = names[cls]
                    
                    mins, secs = divmod(int(time_of_current_frame_sec), 60)
                    hours, mins = divmod(mins, 60)
                    timestamp_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

                    # 🚗 ALPR METRIC: Static roadmap placement for CPU execution
                    detected_plate_text = "N/A"
                    if class_name.lower() in ['car', 'truck', 'bus', 'motorcycle']:
                        detected_plate_text = "Future Work"
                    
                    if class_name.lower() == 'person':
                        if t_id is not None:
                            identity_label = f"Person_{t_id:03d}"
                        else:
                            identity_label = "Person_Initializing"
                        label = f"{identity_label} ({conf:.2f})"
                    else:
                        if t_id is not None:
                            identity_label = f"{class_name.capitalize()}_{t_id:03d}"
                        else:
                            identity_label = class_name.capitalize()
                        
                        if detected_plate_text != "N/A":
                            label = f"{identity_label} | Plate: {detected_plate_text} ({conf:.2f})"
                        else:
                            label = f"{identity_label} ({conf:.2f})"
                    
                    # Log tracking data package directly to the database layer
                    self.db.log_entity_frame(
                        case_id=self.case_id,
                        entity_label=identity_label,
                        class_type=class_name,
                        confidence=float(conf),
                        frame_num=int(frame_counter),
                        video_time=timestamp_str,
                        license_plate=detected_plate_text 
                    )
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                current_detection_count = 0

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
