import os
import csv
import time
import cv2
from PySide6.QtWidgets import QMessageBox

def write_metadata_summary(full_output_dir, video_metadata, evidence_log, last_analysis_rate, filter_settings):
    """Writes the video metadata and analysis filter settings to a text file."""
    metadata_file_path = os.path.join(full_output_dir, "metadata_and_filters.txt")
    
    try:
        with open(metadata_file_path, 'w') as f:
            f.write("="*20 + " VIDEO METADATA " + "="*20 + "\n")
            if video_metadata:
                for key, value in video_metadata.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write("No video metadata available (File not selected or error during extraction).\n")
            
            f.write("\n" + "="*20 + " ANALYSIS FILTER SETTINGS " + "="*20 + "\n")
            f.write(f"Analysis Rate Achieved: {last_analysis_rate}\n")
            for key, value in filter_settings.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
            
        return True
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Could not write metadata summary: {e}")
        return False

def generate_report(evidence_log, output_dir_name, video_metadata, last_analysis_rate):
    """Saves all logged evidence frames to disk and generates a CSV summary report."""
    if not evidence_log:
        QMessageBox.information(None, "No Evidence", "The evidence gallery is empty. Run an analysis first.")
        return None

    # 1. Get and Create Output Directory
    output_dir_name = output_dir_name.strip()
    if not output_dir_name:
        output_dir_name = "video_analysis_output"
    
    # Use a timestamp to ensure uniqueness and prevent overwrites
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir_name, timestamp)

    try:
        os.makedirs(full_output_dir, exist_ok=True)
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Could not create output directory: {e}")
        return None
            
    # 2. Prepare Report Data and Save Frames
    csv_file_path = os.path.join(full_output_dir, "analysis_report.csv")
    image_dir = os.path.join(full_output_dir, "frames")
    os.makedirs(image_dir, exist_ok=True)
    
    report_data = []
    
    # Determine filter header fields dynamically from the first entry
    filter_headers = list(evidence_log[0]['filters_used'].keys())
    header = ["Image_File", "Frame_Number", "Timestamp", "Total_Detections"] + filter_headers
    
    # Save Frames and Gather CSV Data
    for i, entry in enumerate(evidence_log):
        frame_num = entry['frame_num']
        timestamp_hms = entry['timestamp']
        detection_count = entry['detection_count']
        
        # Convert frame from RGB (logged) back to BGR (OpenCV standard) for saving
        frame_bgr = cv2.cvtColor(entry['frame_rgb'], cv2.COLOR_RGB2BGR) 
        
        # Generate filename: TIMESTAMP_FRAMENUMBER.png
        img_filename = f"{timestamp_hms.replace(':', '_')}_F{frame_num}.png"
        img_path = os.path.join(image_dir, img_filename)
        
        # Save the image
        cv2.imwrite(img_path, frame_bgr)
        
        # Create the CSV row
        row = [img_filename, frame_num, timestamp_hms, detection_count]
        # Add filter settings values
        row.extend(list(entry['filters_used'].values()))
        report_data.append(row)

    # 3. Write CSV Report
    try:
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(report_data)
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Could not write CSV report: {e}")
        return None
            
    # 4. Write Metadata Summary File (NEW STEP)
    if evidence_log:
        filter_settings = evidence_log[0]['filters_used']
    else:
        filter_settings = {}
    metadata_success = write_metadata_summary(full_output_dir, video_metadata, evidence_log, last_analysis_rate, filter_settings)
    
    if not metadata_success:
         # If metadata write failed, inform but continue with the main success message
         QMessageBox.warning(None, "Partial Success", 
                            f"Report and {len(evidence_log)} frames saved successfully, but the Metadata Summary file failed to write.")
         return full_output_dir
             
    # 5. Final Success Message
    QMessageBox.information(None, "Success", 
                            f"Complete Analysis Package Saved Successfully! 🎉\n\nOutput folder created: {full_output_dir}\n\nThe package contains:\n- metadata_and_filters.txt (Video Metadata & Analysis Settings)\n- analysis_report.csv (Evidence Log)\n- frames/ (Annotated Images)")
    return full_output_dir
