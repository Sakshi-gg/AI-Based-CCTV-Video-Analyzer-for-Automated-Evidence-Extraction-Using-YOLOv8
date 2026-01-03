def hms_to_seconds(hms_str):
    """Converts HH:MM:SS string to total seconds. Returns -1 on invalid format."""
    try:
        parts = hms_str.split(':')
        if len(parts) != 3:
            return -1
        
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
        
        if m >= 60 or s >= 60 or h < 0 or m < 0 or s < 0:
            return -1

        return h * 3600 + m * 60 + s
    except ValueError:
        return -1

def seconds_to_min_sec_string(seconds):
    """Converts total seconds into 'M min S secs' format."""
    seconds = int(seconds)
    m = seconds // 60
    s = seconds % 60
    
    # Handle the case where minutes is 0 (e.g., "50 secs")
    if m == 0:
        return f"{s} secs"
    # Handle the general case (e.g., "1 min 11 secs")
    return f"{m} min {s} secs"

def seconds_to_hms(seconds):
    """Converts total seconds into HH:MM:SS format."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

