import os
import json
import csv
import openai
from typing import List, Dict, Any, Tuple, Optional, Union

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gpt_assisted_mapping(file_path: str, raw_data: Optional[List[Dict[str, Any]]] = None, headers: Optional[List[str]] = None) -> Tuple[str, str, float, List[Dict[str, Any]]]:
    """
    Enhanced GPT fallback that uses provided data instead of re-reading files
    """
    
    # Generate sample content from provided data
    sample_content = generate_sample_content(raw_data, headers, file_path)
    
    prompt = f"""
You are a medical device log parser expert.
This is a raw exported log file from a medical device (e.g., infusion pump, ventilator, dialysis machine).

ANALYSIS TARGET:
File name: {os.path.basename(file_path)}
Headers: {headers if headers else 'Not available'}
Sample content: {sample_content}

TASK:
1. Identify the most likely device manufacturer and model
2. Map the raw column headers to this standard schema:
   ['timestamp', 'flow_rate', 'motor_current', 'temperature', 'vibration', 'battery_voltage']

KNOWN DEVICES:
- BD Alaris infusion pumps (headers often include: time, temp, flowrate, motorcurrent, battv)
- Baxter Sigma pumps (headers often include: timestamp, infusion_rate, pressure, volume)
- Philips IntelliVue monitors (headers often include: timestamp, heart_rate, blood_pressure, spo2)
- Drager ventilators (headers often include: timestamp, respiratory_rate, tidal_volume, peep)

Respond in JSON format:
{{
  "brand": "BD",
  "device_type": "alaris_8015",
  "confidence": 0.85,
  "mapping": {{
    "Time": "timestamp",
    "Temp(F)": "temperature",
    "FlowRate_mLhr": "flow_rate",
    "MotorCurrent_mA": "motor_current",
    "BattV": "battery_voltage"
  }},
  "reasoning": "Filename contains 'alaris', temperature values suggest Fahrenheit, flow rates in mL/hr pattern"
}}
"""

    content_text = "No content received"  # Initialize with default value
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise, technical assistant for medical device data parsing. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.1  # Low temperature for consistent results
        )

        content_text = response.choices[0].message.content or "No message content"

        # Clean up response (remove markdown code blocks if present)
        content_text = content_text.strip()
        if content_text.startswith("```json"):
            content_text = content_text[7:]
        if content_text.endswith("```"):
            content_text = content_text[:-3]
        content_text = content_text.strip()

        parsed = json.loads(content_text)
        brand = parsed.get("brand", "unknown")
        device_type = parsed.get("device_type", "unknown")
        confidence = float(parsed.get("confidence", 0.3))
        mapping = parsed.get("mapping", {})

        # Apply the mapping to the raw data
        standardized_rows = apply_gpt_mapping(raw_data or [], mapping)

        return brand, device_type, confidence, standardized_rows

    except json.JSONDecodeError as e:
        print(f"GPT response parsing failed: {e}")
        print(f"Response content: {content_text}")
        return "unknown", "unknown", 0.0, raw_data or []
    
    except Exception as e:
        print(f"GPT fallback failed: {e}")
        print(f"Response content: {content_text}")
        return "unknown", "unknown", 0.0, raw_data or []

def generate_sample_content(raw_data: Optional[List[Dict[str, Any]]], headers: Optional[List[str]], file_path: str) -> str:
    """Generate sample content from raw data for GPT analysis"""
    
    if raw_data:
        # Use first few rows of actual data
        sample_rows = raw_data[:3]
        sample_text = f"Headers: {list(sample_rows[0].keys()) if sample_rows else headers}\n"
        sample_text += "Sample rows:\n"
        for i, row in enumerate(sample_rows):
            row_sample = {k: v for k, v in list(row.items())[:6]}  # First 6 fields
            sample_text += f"Row {i+1}: {row_sample}\n"
        return sample_text
    
    elif headers:
        return f"Headers only: {headers}"
    
    else:
        # Fallback: try to read a small sample from file
        try:
            if file_path.endswith(".csv"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read(1000)  # First 1000 characters
            elif file_path.endswith(".json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(2000)  # First 2000 characters
                    return content
        except Exception as e:
            return f"Could not read file sample: {e}"
    
    return "No data available for analysis"

def apply_gpt_mapping(raw_data: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Apply GPT-provided mapping to raw data"""
    
    if not raw_data or not mapping:
        return raw_data
    
    standardized_rows: List[Dict[str, Any]] = []
    
    for row in raw_data:
        new_row: Dict[str, Any] = {}
        
        # Apply mapping
        for raw_key, std_key in mapping.items():
            if raw_key in row:
                new_row[std_key] = row[raw_key]
        
        # Include unmapped fields as-is
        mapped_keys = set(mapping.keys())
        mapped_values = set(mapping.values())
        
        for key, value in row.items():
            if key not in mapped_keys and key not in mapped_values:
                new_row[key] = value
        
        standardized_rows.append(new_row)
    
    return standardized_rows

def enhanced_gpt_assisted_mapping(
    file_path: str, 
    raw_data: Optional[List[Dict[str, Any]]] = None, 
    headers: Optional[List[str]] = None,
    json_context: Optional[Dict[str, Any]] = None
) -> Tuple[str, str, float, List[Dict[str, Any]]]:
    """GPT fallback enhanced with JSON device specifications"""
    
    # Handle None inputs
    if raw_data is None:
        raw_data = []
    if headers is None:
        headers = []
    
    # Build enhanced prompt with JSON context
    json_context_str = ""
    if json_context:
        # Format the JSON context for GPT
        available_devices = {}
        for device_type, devices in json_context.items():
            available_devices[device_type] = []
            for device_key in devices:
                available_devices[device_type].append(device_key)
        
        json_context_str = f"""
DEVICE SPECIFICATION CONTEXT:
Available device specifications:
{json.dumps(available_devices, indent=2)}

This helps identify devices based on known patterns and error code formats.
"""
    
    # Enhanced prompt with JSON awareness
    enhanced_prompt = f"""
You are a medical device log parser expert with access to manufacturer specifications.

{json_context_str}

ANALYSIS TARGET:
File: {os.path.basename(file_path)}
Headers: {headers if headers else 'Not available'}
Sample content: {generate_sample_content(raw_data, headers, file_path)}

ENHANCED DEVICE IDENTIFICATION:
Consider these device-specific patterns:
- BD Alaris: Error codes like "100.1130", separate time/date fields, headers: time, temp, flowrate, motorcurrent, battv
- Baxter Sigma: Simple numeric error codes, IrDA logs, headers: timestamp, infusion_rate, pressure, volume  
- Philips IntelliVue: Error codes like "ALM001", RDE export, headers: timestamp, heart_rate, blood_pressure, spo2
- Drager Narkomed: Field separator "~", 3-digit error codes, headers: timestamp, respiratory_rate, tidal_volume, peep

TASK:
1. Identify the most likely device manufacturer and model
2. Map headers to standard schema: ['timestamp', 'flow_rate', 'motor_current', 'temperature', 'vibration', 'battery_voltage']
3. Provide reasoning based on specific patterns found

Respond in JSON format:
{{
  "brand": "BD",
  "device_type": "alaris_8015",
  "confidence": 0.92,
  "reasoning": "Error codes match BD XXX.XXXN format, separate time/date fields detected, temperature values in Fahrenheit range",
  "mapping": {{
    "Time": "timestamp",
    "Temp(F)": "temperature",
    "FlowRate_mLhr": "flow_rate",
    "MotorCurrent_mA": "motor_current",
    "BattV": "battery_voltage"
  }}
}}
"""

    # Use the same GPT calling logic as your original function
    content_text = "No content received"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise, technical assistant for medical device data parsing. Always respond with valid JSON. Use your knowledge of medical device patterns and the provided context."},
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=1000,  # Increased for more detailed reasoning
            temperature=0.1
        )

        content_text = response.choices[0].message.content or "No message content"

        # Same parsing logic as original
        content_text = content_text.strip()
        if content_text.startswith("```json"):
            content_text = content_text[7:]
        if content_text.endswith("```"):
            content_text = content_text[:-3]
        content_text = content_text.strip()

        parsed = json.loads(content_text)
        brand = parsed.get("brand", "unknown")
        device_type = parsed.get("device_type", "unknown") 
        confidence = float(parsed.get("confidence", 0.3))
        mapping = parsed.get("mapping", {})
        reasoning = parsed.get("reasoning", "No reasoning provided")

        # Apply the mapping
        standardized_rows = apply_gpt_mapping(raw_data, mapping)

        print(f"🔍 Enhanced GPT Analysis: {reasoning}")  # Debug output
        return brand, device_type, confidence, standardized_rows

    except json.JSONDecodeError as e:
        print(f"❌ Enhanced GPT response parsing failed: {e}")
        print(f"Response content: {content_text}")
        # Fallback to original function
        return gpt_assisted_mapping(file_path, raw_data, headers)
    
    except Exception as e:
        print(f"❌ Enhanced GPT fallback failed: {e}")
        # Fallback to original function  
        return gpt_assisted_mapping(file_path, raw_data, headers)

def test_enhanced_vs_original(file_path: str, raw_data: Optional[List[Dict[str, Any]]], headers: Optional[List[str]]):
    """Compare original vs enhanced GPT performance"""
    
    print("\n" + "="*50)
    print("🧪 TESTING GPT ENHANCEMENT")
    print("="*50)
    
    # Handle None inputs
    if raw_data is None:
        raw_data = []
    if headers is None:
        headers = []
    
    # Test original
    print("\n--- 📊 Original GPT Fallback ---")
    try:
        orig_brand, orig_type, orig_conf, _ = gpt_assisted_mapping(file_path, raw_data, headers)
        print(f"✅ Original Result: {orig_brand} {orig_type} (confidence: {orig_conf})")
    except Exception as e:
        print(f"❌ Original failed: {e}")
        orig_conf = 0.0
        orig_brand, orig_type = "error", "error"
    
    # Test enhanced
    print("\n--- 🚀 Enhanced GPT Fallback ---")
    try:
        from device_knowledge import get_all_available_devices
        json_context = get_all_available_devices()
        
        enh_brand, enh_type, enh_conf, _ = enhanced_gpt_assisted_mapping(
            file_path, raw_data, headers, json_context
        )
        print(f"✅ Enhanced Result: {enh_brand} {enh_type} (confidence: {enh_conf})")
        
        # Calculate improvement
        improvement = enh_conf - orig_conf
        print(f"\n📈 Confidence improvement: {improvement:+.2f}")
        
        if improvement > 0:
            print("🎯 Enhanced version performed better!")
        elif improvement == 0:
            print("🤝 Both versions performed equally")
        else:
            print("📉 Original version performed better")
            
    except Exception as e:
        print(f"❌ Enhanced failed: {e}")
    
    print("="*50)


def quick_test_with_sample_data():
    """Quick test with sample medical device data"""
    
    print("🔬 Running Quick Test with Sample Data")
    
    # Sample BD Alaris data
    sample_headers = ["Time", "Date", "Temp(F)", "FlowRate_mLhr", "MotorCurrent_mA", "BattV"]
    sample_data = [
        {
            "Time": "14:30:25",
            "Date": "2024-01-15", 
            "Temp(F)": "98.6",
            "FlowRate_mLhr": "50.0",
            "MotorCurrent_mA": "2500",
            "BattV": "7.2"
        },
        {
            "Time": "14:30:30",
            "Date": "2024-01-15",
            "Temp(F)": "98.8", 
            "FlowRate_mLhr": "50.1",
            "MotorCurrent_mA": "2480",
            "BattV": "7.2"
        }
    ]
    
    test_enhanced_vs_original("alaris_8015_log.csv", sample_data, sample_headers)
