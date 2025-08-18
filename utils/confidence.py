"""
Enhanced Confidence Scoring System V2
Multi-factor device identification with JSON-enhanced patterns
"""

import re
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Try to import fuzzywuzzy for fuzzy string matching
# At the very top of confidence.py, paste this:

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    # Create a tiny stub so fuzz.ratio(a, b) always exists
    class _FuzzStub:
        @staticmethod
        def ratio(a: str, b: str) -> int:
            return 0

    fuzz = _FuzzStub()
    FUZZY_AVAILABLE = False
    print("⚠️ fuzzywuzzy not available – fuzzy matching will always score 0")


class EnhancedConfidenceCalculator:
    def __init__(self, json_specs_dir: Optional[str] = None):
        if json_specs_dir is None:
            json_specs_dir = os.path.join(os.path.dirname(__file__), "data", "device_specs")
        
        self.json_specs_dir = Path(json_specs_dir)
        self.device_signatures = {}
        self._load_device_signatures()
    
    def _load_device_signatures(self):
        """Load device signatures from JSON specifications"""
        try:
            from device_knowledge import get_all_available_devices, get_device_specification_json
            
            available_devices = get_all_available_devices()
            
            for device_type, device_list in available_devices.items():
                for device_key in device_list:
                    spec = get_device_specification_json(device_type, device_key)
                    if spec:
                        signature = self._build_signature_from_json(device_key, device_type, spec)
                        self.device_signatures[device_key] = signature
            
            print(f"✅ Loaded {len(self.device_signatures)} device signatures")
            
        except ImportError as e:
            print(f"⚠️  Could not load JSON specifications: {e}")
            self._load_fallback_signatures()
    
    def _build_signature_from_json(self, device_key: str, device_type: str, spec: Dict) -> Dict:
        """Build device signature from JSON specification"""
        
        signature = {
            "device_key": device_key,
            "device_type": device_type,
            "brand": spec.get("device_info", {}).get("manufacturer", "unknown"),
            "required_headers": spec.get("required_headers", []),
            "optional_headers": spec.get("optional_headers", []),
            "header_variations": spec.get("header_variations", {}),
            "filename_patterns": spec.get("identification_patterns", {}).get("filename_patterns", []),
            "data_validation_ranges": spec.get("data_validation_ranges", {}),
            "error_code_patterns": {},
            "timestamp_patterns": []
        }
        
        # Extract error code patterns
        error_codes = spec.get("error_codes", {})
        if isinstance(error_codes, dict):
            if "format" in error_codes:
                signature["error_code_patterns"]["format_regex"] = self._convert_format_to_regex(error_codes["format"])
            
            # Look for actual error code examples
            codes = error_codes.get("codes", {})
            if codes:
                example_codes = list(codes.keys())[:3]  # First 3 examples
                signature["error_code_patterns"]["examples"] = example_codes
        
        # Extract timestamp patterns based on device specifics
        if "bd" in device_key.lower() and "alaris" in device_key.lower():
            signature["timestamp_patterns"] = ["separate_time_date"]
        
        # Add data format patterns
        log_format = spec.get("log_file_format", {})
        if "field_separator" in log_format:
            signature["field_separator"] = log_format["field_separator"]
        
        return signature
    
    def _convert_format_to_regex(self, format_str: str) -> str:
        """Convert format description to regex pattern"""
        format_mappings = {
            "XXX.XXXN": r"\d{3}\.\d{4}",
            "System codes": r"[A-Z]{3}\d{3}",
            "Decimal and Hexadecimal codes": r"\d{1,3}",
            "Three-digit subsystem code": r"\d{3}\.\d{4}"
        }
        
        return format_mappings.get(format_str, r"\w+")
    
    def _load_fallback_signatures(self):
        """Fallback signatures if JSON loading fails"""
        self.device_signatures = {
            "bd_alaris": {
                "device_key": "bd_alaris",
                "device_type": "infusion_pumps",
                "brand": "BD",
                "required_headers": ["time", "temp", "flowrate", "motorcurrent", "battv"],
                "header_variations": {
                    "time": ["timestamp", "datetime"],
                    "temp": ["temperature", "temp_f"],
                    "flowrate": ["flow_rate", "ml_hr"],
                    "motorcurrent": ["motor_current", "current_ma"],
                    "battv": ["battery_voltage", "voltage"]
                },
                "filename_patterns": ["alaris", "bd", "8015", "8100"],
                "error_code_patterns": {"format_regex": r"\d{3}\.\d{4}"},
                "timestamp_patterns": ["separate_time_date"]
            }
        }
    
    def compute_enhanced_confidence(
        self, 
        headers: List[str], 
        sample_data: List[Dict[str, Any]], 
        filename: str
    ) -> Tuple[str, str, str, float]:
        """Multi-factor confidence calculation with JSON-enhanced patterns"""
        
        best_device = None
        best_confidence = 0.0
        best_device_type = "unknown"
        best_brand = "unknown"
        confidence_details = {}
        
        for device_key, signature in self.device_signatures.items():
            confidence = self._calculate_device_confidence(
                headers, sample_data, filename, signature
            )
            
            confidence_details[device_key] = confidence
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_device = device_key
                best_brand = signature.get("brand", "unknown")
                best_device_type = signature.get("device_type", "unknown")
        
        # Debug output
        print(f"🔍 Confidence Analysis:")
        for device, conf in sorted(confidence_details.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"   {device}: {conf:.3f}")
        
        return best_brand, best_device or "unknown", best_device_type, best_confidence

    
    def _calculate_device_confidence(
        self, 
        headers: List[str], 
        sample_data: List[Dict[str, Any]], 
        filename: str, 
        signature: Dict[str, Any]
    ) -> float:
        """Calculate confidence using multiple factors"""
        
        scores = {}
        
        # Factor 1: Header Similarity (35% weight)
        scores['header_score'] = self._calculate_header_similarity(
            headers, signature
        ) * 0.35
        
        # Factor 2: Error Code Pattern Matching (25% weight)  
        scores['error_code_score'] = self._check_error_code_patterns(
            sample_data, signature
        ) * 0.25
        
        # Factor 3: Filename Pattern Matching (20% weight)
        scores['filename_score'] = self._check_filename_patterns(
            filename, signature
        ) * 0.20
        
        # Factor 4: Data Range Validation (10% weight)
        scores['data_range_score'] = self._validate_data_ranges(
            sample_data, signature
        ) * 0.10
        
        # Factor 5: Timestamp Format Matching (10% weight)
        scores['timestamp_score'] = self._check_timestamp_patterns(
            sample_data, signature
        ) * 0.10
        
        total_confidence = sum(scores.values())
        
        # Debug output for top candidate
        if total_confidence > 0.5:
            print(f"   🎯 {signature['device_key']}: H={scores['header_score']:.2f} E={scores['error_code_score']:.2f} F={scores['filename_score']:.2f} D={scores['data_range_score']:.2f} T={scores['timestamp_score']:.2f}")
        
        return min(total_confidence, 1.0)
    
    def _calculate_header_similarity(self, headers: List[str], signature: Dict) -> float:
        """Enhanced header matching with fuzzy string matching and synonyms"""
        
        required_headers = signature.get("required_headers", [])
        header_variations = signature.get("header_variations", {})
        
        if not required_headers:
            return 0.0
        
        headers_lower = [h.lower().strip() for h in headers]
        matches = 0.0
        
        for required in required_headers:
            required_lower = required.lower()
            
            # Exact match (full points)
            if required_lower in headers_lower:
                matches += 1.0
                continue
            
            # Check variations from JSON (full points)
            variations = header_variations.get(required, [])
            variation_match = False
            for variation in variations:
                if variation.lower() in headers_lower:
                    matches += 1.0
                    variation_match = True
                    break
            
            if variation_match:
                continue
            
            # Fuzzy matching for typos/variations (partial points)
            if FUZZY_AVAILABLE:
                best_fuzzy_score = 0
                for header in headers_lower:

                    fuzzy_score = fuzz.ratio(required_lower, header) / 100.0
                    if fuzzy_score > best_fuzzy_score:
                        best_fuzzy_score = fuzzy_score
                
                # Accept fuzzy matches above 80% (partial credit)
                if best_fuzzy_score >= 0.8:
                    matches += best_fuzzy_score * 0.8  # Reduced weight for fuzzy matches
            
            # Substring matching (minimal points)
            else:
                for header in headers_lower:
                    if required_lower in header or header in required_lower:
                        matches += 0.5  # Partial credit
                        break
        
        return matches / len(required_headers)
    
    def _check_error_code_patterns(self, sample_data: List[Dict], signature: Dict) -> float:
        """Check if data contains device-specific error code patterns"""
        
        error_patterns = signature.get("error_code_patterns", {})
        if not error_patterns or not sample_data:
            return 0.0
        
        format_regex = error_patterns.get("format_regex")
        examples = error_patterns.get("examples", [])
        
        if not format_regex and not examples:
            return 0.0
        
        error_codes_found = 0
        total_checks = 0
        
        # Check for error code fields and values
        for row in sample_data[:10]:  # Check first 10 rows
            for key, value in row.items():
                if any(term in key.lower() for term in ['error', 'alarm', 'event', 'code']):
                    total_checks += 1
                    value_str = str(value)
                    
                    # Check regex pattern
                    if format_regex and re.match(format_regex, value_str):
                        error_codes_found += 1
                        continue
                    
                    # Check against examples
                    if examples and value_str in examples:
                        error_codes_found += 1
        
        return (error_codes_found / total_checks) if total_checks > 0 else 0.0
    
    def _check_filename_patterns(self, filename: str, signature: Dict) -> float:
        """Check filename against device-specific patterns"""
        
        filename_patterns = signature.get("filename_patterns", [])
        if not filename_patterns:
            return 0.0
        
        filename_lower = filename.lower()
        matches = 0
        
        for pattern in filename_patterns:
            if pattern.lower() in filename_lower:
                matches += 1
        
        # Return score based on pattern matches (max 1.0)
        return min(matches / len(filename_patterns), 1.0)
    
    def _validate_data_ranges(self, sample_data: List[Dict], signature: Dict) -> float:
        """Validate data values against expected ranges"""
        
        data_ranges = signature.get("data_validation_ranges", {})
        if not data_ranges or not sample_data:
            return 0.0
        
        valid_values = 0
        total_checks = 0
        
        for row in sample_data[:5]:  # Check first 5 rows
            for field, value in row.items():
                field_lower = field.lower()
                
                # Find matching range specification
                range_spec = None
                for range_field, spec in data_ranges.items():
                    if range_field.lower() == field_lower or field_lower in range_field.lower():
                        range_spec = spec
                        break
                
                if range_spec:
                    total_checks += 1
                    try:
                        numeric_value = float(value)
                        min_val = range_spec.get("min", float('-inf'))
                        max_val = range_spec.get("max", float('inf'))
                        
                        if min_val <= numeric_value <= max_val:
                            valid_values += 1
                    except (ValueError, TypeError):
                        pass
        
        return (valid_values / total_checks) if total_checks > 0 else 0.0
    
    def _check_timestamp_patterns(self, sample_data: List[Dict], signature: Dict) -> float:
        """Check timestamp format patterns"""
        
        if not sample_data:
            return 0.0
        
        timestamp_patterns = signature.get("timestamp_patterns", [])
        
        # BD Alaris specific: separate time/date fields
        if "separate_time_date" in timestamp_patterns:
            for row in sample_data[:3]:
                if "time" in row and "date" in row:
                    return 1.0  # Strong indicator for BD Alaris
                # Also check for Time/Date (capitalized)
                if "Time" in row and "Date" in row:
                    return 1.0
        
        # Check for field separator (Drager specific)
        field_separator = signature.get("field_separator")
        if field_separator:
            # This would need to be checked during file reading
            # For now, return neutral score
            return 0.5
        
        # Generic timestamp check
        for row in sample_data[:3]:
            if "timestamp" in row:
                return 0.8  # Good but not device-specific
        
        return 0.0

# Global instance for easy access
_confidence_calculator = None

def get_confidence_calculator() -> EnhancedConfidenceCalculator:
    """Get global confidence calculator instance"""
    global _confidence_calculator
    if _confidence_calculator is None:
        _confidence_calculator = EnhancedConfidenceCalculator()
    return _confidence_calculator

# Enhanced main function
def compute_enhanced_confidence(
    headers: List[str], 
    sample_data: List[Dict[str, Any]], 
    filename: str
) -> Tuple[str, str, str, float]:
    """Enhanced confidence calculation entry point"""
    calculator = get_confidence_calculator()
    return calculator.compute_enhanced_confidence(headers, sample_data, filename)

# Backward compatibility function - KEEP THIS
def compute_confidence(headers: list, expected_fields: list) -> float:
    """Backward compatibility - enhanced version"""
    if not headers or not expected_fields:
        return 0.0
    
    # Use enhanced header similarity calculation
    calculator = get_confidence_calculator()
    
    # Create a simple signature for backward compatibility
    signature = {
        "required_headers": expected_fields,
        "header_variations": {},
        "error_code_patterns": {},
        "filename_patterns": [],
        "data_validation_ranges": {}
    }
    
    return calculator._calculate_header_similarity(headers, signature)

# Test function
def test_enhanced_confidence():
    """Test the enhanced confidence calculator"""
    print("🧪 Testing Enhanced Confidence Calculator")
    
    # Sample BD Alaris data
    headers = ["Time", "Date", "Temp(F)", "FlowRate_mLhr", "MotorCurrent_mA", "BattV"]
    sample_data = [
        {
            "Time": "14:30:25",
            "Date": "2024-01-15", 
            "Temp(F)": "98.6",
            "FlowRate_mLhr": "50.0",
            "MotorCurrent_mA": "2500",
            "BattV": "7.2"
        }
    ]
    filename = "alaris_8015_log.csv"
    
    brand, device, device_type, confidence = compute_enhanced_confidence(
        headers, sample_data, filename
    )
    
    print(f"✅ Result: {brand} {device} ({device_type}) - Confidence: {confidence:.3f}")

if __name__ == "__main__":
    test_enhanced_confidence()
