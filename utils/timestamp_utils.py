"""
Enhanced Timestamp Utilities with Error Tracking and Recovery
"""

from datetime import datetime
from dateutil import parser as date_parser
from typing import List, Dict, Any, Tuple, Optional, Union
import re

def parse_and_sort_timestamps_enhanced(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Enhanced timestamp parsing with error tracking and recovery
    """
    warnings = []
    valid_rows = []
    invalid_rows = []
    
    def try_parse_timestamp(ts_value: Any, row_idx: int) -> Tuple[Optional[datetime], Optional[str]]:
        """Try multiple parsing strategies"""
        
        if not ts_value:
            return None, f"Row {row_idx}: Empty timestamp"
        
        # Convert to string if not already
        ts_str = str(ts_value).strip()
        
        if not ts_str:
            return None, f"Row {row_idx}: Empty timestamp string"
        
        # Strategy 1: dateutil parser (most flexible)
        try:
            parsed_dt = date_parser.parse(ts_str)
            return parsed_dt, None
        except Exception:
            pass
        
        # Strategy 2: Common medical device timestamp formats
        common_formats = [
            "%Y-%m-%d %H:%M:%S",      # 2024-01-15 14:30:25
            "%m/%d/%Y %H:%M:%S",      # 01/15/2024 14:30:25
            "%d/%m/%Y %H:%M:%S",      # 15/01/2024 14:30:25
            "%Y%m%d_%H%M%S",          # 20240115_143025
            "%Y-%m-%dT%H:%M:%S",      # ISO format without timezone
            "%Y-%m-%dT%H:%M:%SZ",     # ISO format with Z
            "%Y-%m-%d %H:%M",         # Without seconds
            "%m/%d/%Y %H:%M",         # US format without seconds
        ]
        
        for fmt in common_formats:
            try:
                parsed_dt = datetime.strptime(ts_str, fmt)
                return parsed_dt, None
            except ValueError:
                continue
        
        # Strategy 3: Extract timestamp from complex strings
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',  # YYYY-MM-DD HH:MM:SS
            r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})',  # MM/DD/YYYY HH:MM:SS
            r'(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2})',      # YYYYMMDD_HHMMSS
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, ts_str)
            if match:
                try:
                    extracted = match.group(1)
                    # Try parsing the extracted timestamp
                    for fmt in common_formats:
                        try:
                            parsed_dt = datetime.strptime(extracted, fmt)
                            return parsed_dt, f"Row {row_idx}: Extracted timestamp from '{ts_str}'"
                        except ValueError:
                            continue
                except Exception:
                    pass
        
        return None, f"Row {row_idx}: Could not parse timestamp '{ts_str}'"
    
    # Process each row
    for i, row in enumerate(rows):
        if "timestamp" not in row:
            warnings.append(f"Row {i}: No timestamp field found")
            invalid_rows.append(row)
            continue
        
        original_ts = row["timestamp"]
        parsed_ts, warning = try_parse_timestamp(original_ts, i)
        
        if parsed_ts:
            # Successful parsing
            row_copy = row.copy()
            row_copy["timestamp"] = parsed_ts
            row_copy["_original_timestamp"] = original_ts  # Keep original for reference
            valid_rows.append(row_copy)
            
            if warning:  # Extraction warning
                warnings.append(warning)
        else:
            # Failed parsing
            if warning:
                warnings.append(warning)
            row["timestamp_parse_error"] = True
            row["_original_timestamp"] = original_ts
            invalid_rows.append(row)
    
    # Report parsing results
    if invalid_rows:
        warnings.append(f"Failed to parse {len(invalid_rows)} timestamps out of {len(rows)} total rows")
    
    if not valid_rows:
        warnings.append("No valid timestamps found - cannot sort data chronologically")
        return rows, warnings  # Return original data if no valid timestamps
    
    # Sort valid rows by timestamp
    try:
        valid_rows.sort(key=lambda r: r["timestamp"])
        warnings.append(f"Successfully sorted {len(valid_rows)} rows chronologically")
    except Exception as e:
        warnings.append(f"Error sorting by timestamp: {e}")
    
    # Convert timestamps back to ISO format strings
    for row in valid_rows:
        try:
            row["timestamp"] = row["timestamp"].isoformat()
        except Exception as e:
            warnings.append(f"Error converting timestamp to ISO format: {e}")
    
    # Include invalid rows at the end (optional - for data completeness)
    # Uncomment if you want to keep invalid timestamp rows
    # valid_rows.extend(invalid_rows)
    
    return valid_rows, warnings

def validate_timestamp_sequence(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate timestamp sequence for anomalies"""
    
    warnings = []
    
    if len(rows) < 2:
        return rows, warnings
    
    # Check for timestamp anomalies
    timestamps = []
    for i, row in enumerate(rows):
        if "timestamp" in row:
            try:
                if isinstance(row["timestamp"], str):
                    ts = date_parser.parse(row["timestamp"])
                else:
                    ts = row["timestamp"]
                timestamps.append((i, ts))
            except Exception:
                pass
    
    if len(timestamps) < 2:
        return rows, warnings
    
    # Check for large gaps
    time_gaps = []
    for i in range(1, len(timestamps)):
        prev_idx, prev_ts = timestamps[i-1]
        curr_idx, curr_ts = timestamps[i]
        gap = (curr_ts - prev_ts).total_seconds()
        time_gaps.append(gap)
    
    if time_gaps:
        avg_gap = sum(time_gaps) / len(time_gaps)
        max_gap = max(time_gaps)
        
        # Alert on unusually large gaps (10x average)
        if max_gap > avg_gap * 10 and avg_gap > 0:
            warnings.append(f"Large time gap detected: {max_gap:.1f}s (avg: {avg_gap:.1f}s)")
        
        # Check for reverse timestamps
        negative_gaps = [gap for gap in time_gaps if gap < 0]
        if negative_gaps:
            warnings.append(f"Found {len(negative_gaps)} reverse timestamp sequences")
    
    return rows, warnings

def infer_timestamp_frequency(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Infer the sampling frequency from timestamps"""
    
    if len(rows) < 3:
        return {"frequency": "unknown", "intervals": []}
    
    intervals = []
    
    for i in range(1, min(20, len(rows))):  # Check first 20 rows
        try:
            if "timestamp" in rows[i] and "timestamp" in rows[i-1]:
                ts1 = date_parser.parse(rows[i-1]["timestamp"]) if isinstance(rows[i-1]["timestamp"], str) else rows[i-1]["timestamp"]
                ts2 = date_parser.parse(rows[i]["timestamp"]) if isinstance(rows[i]["timestamp"], str) else rows[i]["timestamp"]
                
                interval = (ts2 - ts1).total_seconds()
                if interval > 0:
                    intervals.append(interval)
        except Exception:
            continue
    
    if not intervals:
        return {"frequency": "unknown", "intervals": []}
    
    avg_interval = sum(intervals) / len(intervals)
    
    # Classify frequency
    if avg_interval <= 1:
        frequency = "high_frequency"  # Sub-second or 1 second
    elif avg_interval <= 10:
        frequency = "medium_frequency"  # Every few seconds
    elif avg_interval <= 60:
        frequency = "low_frequency"  # Every minute or less
    else:
        frequency = "very_low_frequency"  # Greater than 1 minute
    
    return {
        "frequency": frequency,
        "avg_interval_seconds": round(avg_interval, 2),
        "intervals": intervals[:10],  # First 10 intervals
        "total_samples": len(intervals)
    }

# Backward compatibility - keep original function
def parse_and_sort_timestamps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Original function for backward compatibility"""
    def try_parse(ts: Any) -> Optional[datetime]:
        try:
            return date_parser.parse(str(ts))
        except Exception:
            return None

    for row in rows:
        if "timestamp" in row:
            parsed_ts = try_parse(row["timestamp"])
            row["timestamp"] = parsed_ts if parsed_ts else row["timestamp"]

    # Filter out any rows that failed parsing
    valid_rows = [r for r in rows if isinstance(r["timestamp"], datetime)]

    # Sort by timestamp
    valid_rows.sort(key=lambda r: r["timestamp"])

    # Convert back to ISO format
    for row in valid_rows:
        row["timestamp"] = row["timestamp"].isoformat()

    return valid_rows

def get_timestamp_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get comprehensive timestamp statistics"""
    
    if not rows:
        return {"total_rows": 0}
    
    timestamp_info = {
        "total_rows": len(rows),
        "rows_with_timestamps": 0,
        "earliest_timestamp": None,
        "latest_timestamp": None,
        "duration_seconds": 0,
        "frequency_analysis": {}
    }
    
    valid_timestamps = []
    
    for row in rows:
        if "timestamp" in row:
            timestamp_info["rows_with_timestamps"] += 1
            try:
                ts = date_parser.parse(row["timestamp"]) if isinstance(row["timestamp"], str) else row["timestamp"]
                valid_timestamps.append(ts)
            except Exception:
                pass
    
    if valid_timestamps:
        valid_timestamps.sort()
        timestamp_info["earliest_timestamp"] = valid_timestamps[0].isoformat()
        timestamp_info["latest_timestamp"] = valid_timestamps[-1].isoformat()
        timestamp_info["duration_seconds"] = (valid_timestamps[-1] - valid_timestamps[0]).total_seconds()
        timestamp_info["frequency_analysis"] = infer_timestamp_frequency(rows)
    
    return timestamp_info
