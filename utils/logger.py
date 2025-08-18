import os
import traceback
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

def log_error(file_path: str, exception: Exception, error_dir: str) -> None:
    """Original error logging function for backward compatibility"""
    os.makedirs(error_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(file_path)
    error_filename = f"error_{base_name}_{timestamp}.log"
    error_path = os.path.join(error_dir, error_filename)

    with open(error_path, 'w', encoding='utf-8') as f:
        f.write(f"Error parsing file: {file_path}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(traceback.format_exc())

    print(f"[!] Error logged to {error_path}")

def log_error_enhanced(file_path: str, exception: Exception, parsing_result: Dict[str, Any], error_dir: str) -> Optional[str]:
    """Enhanced error logging with parsing context"""
    os.makedirs(error_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(file_path)
    error_filename = f"error_enhanced_{base_name}_{timestamp}.json"
    error_path = os.path.join(error_dir, error_filename)

    error_data = {
        "timestamp": timestamp,
        "file_path": file_path,
        "file_name": base_name,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "traceback": traceback.format_exc(),
        "parsing_context": {
            "processing_log": parsing_result.get("processing_log", []),
            "warnings": parsing_result.get("warnings", []),
            "device_info": parsing_result.get("device_info", {}),
            "confidence": parsing_result.get("confidence", 0.0)
        },
        "system_info": {
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "file_exists": os.path.exists(file_path)
        }
    }

    try:
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, default=str)
        
        print(f"[!] Enhanced error logged to {error_path}")
        
        # Also create a simple text log for quick reading
        text_log_path = error_path.replace('.json', '.log')
        with open(text_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Enhanced Error Log\n")
            f.write(f"==================\n")
            f.write(f"File: {file_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Exception: {type(exception).__name__}: {str(exception)}\n\n")
            f.write(f"Processing Log:\n")
            for log_entry in parsing_result.get("processing_log", []):
                f.write(f"  - {log_entry}\n")
            f.write(f"\nWarnings:\n")
            for warning in parsing_result.get("warnings", []):
                f.write(f"  - {warning}\n")
            f.write(f"\nTraceback:\n{traceback.format_exc()}")
        
        return error_path
        
    except Exception as log_error_exc:
        # Fallback to original logging if enhanced logging fails
        print(f"[!] Enhanced logging failed: {log_error_exc}")
        log_error(file_path, exception, error_dir)
        return None

def get_error_summary(error_dir: str) -> Dict[str, Any]:
    """Get summary of all logged errors"""
    
    if not os.path.exists(error_dir):
        return {"total_errors": 0, "error_files": []}
    
    error_files = [f for f in os.listdir(error_dir) if f.endswith(('.log', '.json'))]
    
    error_summary: Dict[str, Any] = {
        "total_errors": len(error_files),
        "error_files": error_files,
        "recent_errors": [],
        "common_exceptions": {}
    }
    
    # Analyze recent JSON error logs
    json_files = [f for f in error_files if f.endswith('.json')]
    json_files.sort(reverse=True)  # Most recent first
    
    for error_file in json_files[:10]:  # Last 10 errors
        try:
            error_path = os.path.join(error_dir, error_file)
            with open(error_path, 'r', encoding='utf-8') as f:
                error_data = json.load(f)
            
            error_summary["recent_errors"].append({
                "file": error_data.get("file_name", "unknown"),
                "timestamp": error_data.get("timestamp", "unknown"),
                "exception": error_data.get("exception_type", "unknown"),
                "message": error_data.get("exception_message", "")[:100]  # First 100 chars
            })
            
            # Count exception types
            exc_type = error_data.get("exception_type", "Unknown")
            error_summary["common_exceptions"][exc_type] = error_summary["common_exceptions"].get(exc_type, 0) + 1
            
        except Exception:
            pass  # Skip corrupted error logs
    
    return error_summary

def clean_old_error_logs(error_dir: str, days_to_keep: int = 30) -> int:
    """Clean error logs older than specified days"""
    
    if not os.path.exists(error_dir):
        return 0
    
    from datetime import timedelta
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    cleaned_count = 0
    
    for filename in os.listdir(error_dir):
        file_path = os.path.join(error_dir, filename)
        
        try:
            # Get file modification time
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if file_mtime < cutoff_date:
                os.remove(file_path)
                cleaned_count += 1
                
        except Exception:
            pass  # Skip files we can't process
    
    return cleaned_count
