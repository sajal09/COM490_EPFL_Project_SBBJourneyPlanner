DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def time_to_seconds(time_str: str) -> int:
    """Convert "HH:MM" or "HH:MM:SS" to seconds since midnight.
    
    Examples:
        "12:30"    → 45000
        "12:30:45" → 45045
    """
    parts = time_str.strip().split(":")
    hours   = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_time(seconds: int) -> str:
    """Convert seconds since midnight to "HH:MM:SS".
    
    Handles times past midnight (e.g. GTFS allows > 86400).
    
    Examples:
        45000 → "12:30:00"
        90000 → "25:00:00"
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, secs    = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def day_to_column(day: str) -> str:
    """Convert a day name to its GTFS calendar column name.
    
    Examples:
        "monday"  → "monday"
        "Monday"  → "monday"
        "MONDAY"  → "monday"
    
    Raises:
        ValueError: if the day name is not recognised.
    """
    day = day.strip().lower()
    if day not in DAYS:
        raise ValueError(f"Unrecognised day: '{day}'. Expected one of {DAYS}")
    return day
