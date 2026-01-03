import numpy as np
import cv2

def get_hsv_range(color_name):
    """
    Returns a list of (lower_bound, upper_bound) tuples for HSV color ranges.
    OpenCV uses H: 0-179, S: 0-255, V: 0-255.
    """
    color_name = color_name.lower()
    
    # White: High Value (V > 180) and Low Saturation (S < 50)
    if color_name == "white":
        return [(np.array([0, 0, 180]), np.array([180, 50, 255]))]
    
    # Black: Low Value (V < 80) and low Saturation (S < 150). Adjusted for robustness in dark video.
    if color_name == "black":
        return [(np.array([0, 0, 0]), np.array([180, 150, 80]))]
    
    # Red: Split due to the Hue circle (0-10 and 170-180)
    if color_name == "red":
        return [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            (np.array([170, 50, 50]), np.array([180, 255, 255]))
        ]
    
    elif color_name == "blue":
        return [(np.array([100, 50, 50]), np.array([140, 255, 255]))]
    elif color_name == "green":
        return [(np.array([40, 50, 50]), np.array([80, 255, 255]))]
    elif color_name == "yellow":
        return [(np.array([20, 50, 50]), np.array([40, 255, 255]))]
    
    return None 

def is_color_match(image_roi, color_filter_name, match_threshold=0.20):
    """
    Checks if the cropped image region (ROI) contains the target color
    above a specified percentage threshold.
    """
    if color_filter_name.lower() == 'none' or image_roi.size == 0:
        return True

    hsv_ranges = get_hsv_range(color_filter_name)
    if not hsv_ranges:
        return True 

    # Convert the region of interest to HSV
    hsv_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    
    # Create an initial black mask
    total_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    
    # Combine masks for all color ranges (e.g., for Red)
    for lower, upper in hsv_ranges:
        mask = cv2.inRange(hsv_roi, lower, upper)
        total_mask = cv2.bitwise_or(total_mask, mask)

    # Calculate the percentage of pixels that match the target color
    total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
    matching_pixels = cv2.countNonZero(total_mask)
    
    color_percentage = matching_pixels / total_pixels
    
    # Return true if the match percentage exceeds the threshold
    return color_percentage > match_threshold

