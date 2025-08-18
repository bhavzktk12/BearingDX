"""
Enhanced Unit Converter with Device-Specific Knowledge
"""

from typing import List, Dict, Any, Tuple, Optional, Union

def fahrenheit_to_celsius(f: Union[int, float]) -> float:
    """Convert Fahrenheit to Celsius"""
    return (f - 32) * 5.0 / 9.0

def milliamps_to_amps(ma: Union[int, float]) -> float:
    """Convert milliamps to amps"""
    return ma / 1000.0

def psi_to_mmhg(psi: Union[int, float]) -> float:
    """Convert PSI to mmHg"""
    return psi * 51.715

def normalize_units_enhanced(data: List[Dict[str, Any]], device_metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Enhanced unit conversion with device-specific knowledge and logging
    """
    normalized = []
    conversion_log = []
    
    device_key = None
    if device_metadata:
        brand = device_metadata.get("brand", "unknown")
        device_type = device_metadata.get("type", "unknown")
        device_key = f"{brand}_{device_type}"
    
    for row_idx, row in enumerate(data):
        new_row = {}
        
        for key, value in row.items():
            converted_value, conversion_info = smart_unit_conversion(
                key, value, device_key, row_idx
            )
            new_row[key] = converted_value
            
            if conversion_info:
                conversion_log.append(conversion_info)
        
        normalized.append(new_row)
    
    return normalized, conversion_log

def smart_unit_conversion(key: str, value: Any, device_key: Optional[str] = None, row_idx: int = 0) -> Tuple[Any, Optional[str]]:
    """
    Convert units based on device knowledge and intelligent heuristics
    """
    
    # Skip non-numeric values
    if not isinstance(value, (int, float)):
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return value, None
    else:
        numeric_value = value
    
    key_lower = key.lower().strip()
    
    # Device-specific unit conversion (based on our medical device research)
    if device_key:
        conversion_result = apply_device_specific_conversion(
            key_lower, numeric_value, device_key, row_idx
        )
        if conversion_result:
            return conversion_result
    
    # Fallback to heuristic-based conversion
    return apply_heuristic_conversion(key_lower, numeric_value, row_idx)

def apply_device_specific_conversion(key_lower: str, value: Union[int, float], device_key: str, row_idx: int) -> Optional[Tuple[float, str]]:
    """Apply device-specific unit conversions based on known specifications"""
    
    # Import here to avoid circular imports
    try:
        from device_knowledge import DEVICE_KNOWLEDGE
        device_patterns = DEVICE_KNOWLEDGE.get("data_patterns", {}).get(device_key, {})
    except ImportError:
        device_patterns = {}
    
    # BD Alaris specific conversions
    if device_key.startswith("BD_alaris"):
        if "temp" in key_lower:
            # BD Alaris logs temperature in Fahrenheit
            if 60 <= value <= 120:  # Typical Fahrenheit range for medical devices
                converted = round(fahrenheit_to_celsius(value), 2)
                return converted, f"Row {row_idx}: Converted {value}°F to {converted}°C (BD Alaris standard)"
        
        elif "current" in key_lower and ("ma" in key_lower or value > 100):
            # BD Alaris logs motor current in milliamps
            converted = round(milliamps_to_amps(value), 4)
            return converted, f"Row {row_idx}: Converted {value}mA to {converted}A (BD Alaris standard)"
    
    # Baxter Sigma specific conversions
    elif device_key.startswith("Baxter_sigma"):
        if "pressure" in key_lower and "psi" in key_lower:
            # Convert PSI to mmHg for consistency
            converted = round(psi_to_mmhg(value), 1)
            return converted, f"Row {row_idx}: Converted {value}PSI to {converted}mmHg (Baxter standard)"
    
    # Philips IntelliVue specific conversions
    elif device_key.startswith("Philips_intellivue"):
        # Philips typically uses standard medical units, minimal conversion needed
        pass
    
    return None

def apply_heuristic_conversion(key_lower: str, value: Union[int, float], row_idx: int) -> Tuple[Union[int, float], Optional[str]]:
    """Apply heuristic-based unit conversion with warnings"""
    
    # Temperature conversion
    if "temperature" in key_lower or "temp" in key_lower:
        if value > 60:  # Likely Fahrenheit
            converted = round(fahrenheit_to_celsius(value), 2)
            warning = f"Row {row_idx}: ASSUMED {value}°F → {converted}°C (verify unit!)"
            return converted, warning
    
    # Current conversion
    elif "current" in key_lower:
        if value > 100:  # Likely milliamps
            converted = round(milliamps_to_amps(value), 4)
            warning = f"Row {row_idx}: ASSUMED {value}mA → {converted}A (verify unit!)"
            return converted, warning
    
    # Pressure conversion
    elif "pressure" in key_lower:
        if 0.1 <= value <= 50:  # Likely PSI range
            converted = round(psi_to_mmhg(value), 1)
            warning = f"Row {row_idx}: ASSUMED {value}PSI → {converted}mmHg (verify unit!)"
            return converted, warning
    
    # No conversion applied
    return value, None

def validate_converted_values(data: List[Dict[str, Any]], device_key: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate converted values against expected ranges"""
    
    validated_data = []
    warnings = []
    
    # Get expected ranges for device
    expected_ranges = {}
    if device_key:
        try:
            from device_knowledge import DEVICE_KNOWLEDGE
            expected_ranges = DEVICE_KNOWLEDGE.get("data_patterns", {}).get(device_key, {})
        except ImportError:
            pass
    
    for row_idx, row in enumerate(data):
        validated_row = row.copy()
        
        for field, value in row.items():
            if isinstance(value, (int, float)) and field in expected_ranges:
                field_range = expected_ranges[field]
                
                # Check if value is within expected range
                if "min" in field_range and value < field_range["min"]:
                    warnings.append(f"Row {row_idx}: {field} value {value} below expected minimum {field_range['min']}")
                elif "max" in field_range and value > field_range["max"]:
                    warnings.append(f"Row {row_idx}: {field} value {value} above expected maximum {field_range['max']}")
        
        validated_data.append(validated_row)
    
    return validated_data, warnings

# Backward compatibility - keep original function
def normalize_units(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Original normalize_units function for backward compatibility"""
    normalized = []

    for row in data:
        new_row = {}
        for key, value in row.items():
            try:
                # Convert strings to floats where possible
                val = float(value)
                if "temperature" in key.lower() and val > 60:  # likely °F
                    new_row[key] = round(fahrenheit_to_celsius(val), 2)
                elif "current" in key.lower() and val > 100:  # likely mA
                    new_row[key] = round(milliamps_to_amps(val), 4)
                else:
                    new_row[key] = val
            except:
                # Non-numeric values are passed as-is
                new_row[key] = value
        normalized.append(new_row)

    return normalized

def get_unit_conversion_stats(conversion_log: List[str]) -> Dict[str, Any]:
    """Generate statistics about unit conversions performed"""
    
    stats = {
        "total_conversions": len(conversion_log),
        "conversion_types": {},
        "assumptions_made": 0,
        "device_specific": 0
    }
    
    for log_entry in conversion_log:
        if "ASSUMED" in log_entry:
            stats["assumptions_made"] += 1
        if "standard)" in log_entry:
            stats["device_specific"] += 1
        
        # Count conversion types
        if "°F to" in log_entry:
            stats["conversion_types"]["temperature"] = stats["conversion_types"].get("temperature", 0) + 1
        elif "mA to" in log_entry:
            stats["conversion_types"]["current"] = stats["conversion_types"].get("current", 0) + 1
        elif "PSI to" in log_entry:
            stats["conversion_types"]["pressure"] = stats["conversion_types"].get("pressure", 0) + 1
    
    return stats
