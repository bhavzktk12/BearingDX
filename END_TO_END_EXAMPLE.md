# BearingDX End-to-End Processing Example

This document demonstrates the complete workflow from loading a CWRU .mat file to producing a final bearing health diagnosis.

## Example Workflow Implementation

```python
#!/usr/bin/env python3
"""
BearingDX End-to-End Processing Example

This script demonstrates the complete workflow for processing a single bearing file
from raw .mat data to final health classification.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Example data structures (based on INTERFACES.md)
@dataclass
class BearingGeometry:
    rolling_elements: int = 9
    ball_diameter_mm: float = 7.938
    pitch_diameter_mm: float = 39.04
    contact_angle_deg: float = 0.0

@dataclass
class ProcessingConfig:
    window_size: int = 2048
    overlap: float = 0.5
    normalization: str = 'zscore'
    min_file_quality: float = 0.6
    sensor_preference: str = 'DE'

def end_to_end_bearing_diagnosis(file_path: str, config: ProcessingConfig) -> Dict[str, Any]:
    """
    Complete end-to-end bearing diagnosis workflow.
    
    Args:
        file_path: Path to CWRU .mat file
        config: Processing configuration
        
    Returns:
        Complete diagnosis results with health classification
    """
    
    print(f"🔍 Starting BearingDX analysis of: {Path(file_path).name}")
    print("="*60)
    
    # ========================================
    # Step 1: Load and Inspect .mat File
    # ========================================
    print("📁 Step 1: Loading .mat file...")
    
    # This would use bdxio.loader.load_mat()
    signal_data = load_mat_file(file_path, sensor=config.sensor_preference)
    
    print(f"   ✓ Signal loaded: {len(signal_data['signal'])} samples @ {signal_data['fs']} Hz")
    print(f"   ✓ Available sensors: {signal_data['available_sensors']}")
    print(f"   ✓ Primary sensor: {signal_data['sensor']}")
    
    # Infer operating conditions
    load = infer_load_from_filename(Path(file_path).name)
    rpm = get_rpm_from_load(load) if load is not None else 1797
    
    print(f"   ✓ Detected load: {load} HP, RPM: {rpm}")
    
    # ========================================
    # Step 2: Signal Segmentation & Quality Assessment
    # ========================================
    print("\n🔧 Step 2: Signal segmentation and quality assessment...")
    
    # Segment the signal
    segments = segment_signal(
        signal_data['signal'], 
        signal_data['fs'],
        window_size=config.window_size,
        overlap=config.overlap,
        normalization=config.normalization
    )
    
    # Assess quality of each segment
    quality_scores = []
    valid_segments = []
    
    for i, segment in enumerate(segments):
        quality = assess_segment_quality(segment)
        quality_scores.append(quality)
        
        if quality.is_valid:
            valid_segments.append((i, segment, quality))
    
    file_quality = len(valid_segments) / len(segments)
    
    print(f"   ✓ Total segments: {len(segments)}")
    print(f"   ✓ Valid segments: {len(valid_segments)} ({file_quality:.1%})")
    print(f"   ✓ File quality score: {file_quality:.3f}")
    
    if file_quality < config.min_file_quality:
        print(f"   ⚠️  Warning: File quality below threshold ({config.min_file_quality})")
    
    # ========================================
    # Step 3: Time Domain Feature Extraction
    # ========================================
    print("\n📊 Step 3: Time domain feature extraction...")
    
    # Compute TD features for each valid segment
    td_features_list = []
    for seg_idx, segment, quality in valid_segments:
        td_features = compute_td_features(segment)
        td_features_list.append(td_features)
    
    # Compute file-level TD indicators
    td_indicators = compute_file_td_indicators(td_features_list, quality_scores)
    
    print(f"   ✓ Mean kurtosis: {td_indicators.mean_kurtosis:.2f}")
    print(f"   ✓ Max kurtosis: {td_indicators.max_kurtosis:.2f}")
    print(f"   ✓ Kurtosis burst index: {td_indicators.kurtosis_burst_index:.1%}")
    print(f"   ✓ Mean crest factor: {td_indicators.mean_crest_factor:.2f}")
    print(f"   ✓ RMS trend slope: {td_indicators.rms_trend_slope:.6f}")
    
    # Initial TD-based fault screening
    td_fault_suggestion, td_confidence = infer_fault_location_td(td_indicators)
    print(f"   ✓ TD fault suggestion: {td_fault_suggestion} (confidence: {td_confidence:.3f})")
    
    # ========================================
    # Step 4: Frequency Domain Analysis
    # ========================================
    print("\n🌊 Step 4: Frequency domain analysis...")
    
    # Calculate bearing fault frequencies
    bearing_geometry = BearingGeometry()  # CWRU default
    fault_freqs = calculate_fault_frequencies(rpm, bearing_geometry)
    
    print(f"   ✓ Fault frequencies @ {rpm} RPM:")
    print(f"     - BPFI: {fault_freqs.bpfi:.1f} Hz")
    print(f"     - BPFO: {fault_freqs.bpfo:.1f} Hz") 
    print(f"     - BSF:  {fault_freqs.bsf:.1f} Hz")
    print(f"     - FTF:  {fault_freqs.ftf:.1f} Hz")
    
    # Select top quality segments for FD analysis
    n_top_segments = max(1, len(valid_segments) // 4)  # Top 25%
    top_segments = sorted(valid_segments, key=lambda x: x[2].score, reverse=True)[:n_top_segments]
    
    print(f"   ✓ Using top {len(top_segments)} segments for FD analysis")
    
    # Concatenate top segments for envelope analysis
    combined_signal = np.concatenate([seg[1] for seg in top_segments])
    
    # Envelope analysis with automatic band selection
    envelope_results = compute_envelope_analysis(
        combined_signal,
        signal_data['fs'], 
        fault_freqs,
        candidate_bands=[(1000, 3000), (2000, 5000), (3000, 6000), (6000, 10000)]
    )
    
    print(f"   ✓ Selected envelope band: {envelope_results.selected_band[0]:.0f}-{envelope_results.selected_band[1]:.0f} Hz")
    print(f"   ✓ Envelope SNRs:")
    for freq_name, snr in envelope_results.fault_snrs.items():
        print(f"     - {freq_name}: {snr:.2f}")
    
    # FFT-based energy ratios
    fft_ratios = compute_fft_ratios(combined_signal, signal_data['fs'], fault_freqs)
    
    # ========================================
    # Step 5: TD + FD Fusion
    # ========================================
    print("\n🔀 Step 5: Evidence fusion and fault classification...")
    
    # Apply Normal fast-path check (envelope-only)
    normal_gate_result = apply_normal_fast_path(td_indicators, envelope_results)
    
    if normal_gate_result == 'Normal':
        print("   ✓ Passed Normal fast-path gate")
        final_fault_label = 'Normal'
        confidence = 0.85
    else:
        print("   ✓ Proceeding with full fault analysis")
        
        # Full fusion logic
        fusion_result = fuse_td_fd_evidence(
            td_fault_suggestion=td_fault_suggestion,
            td_confidence=td_confidence,
            envelope_results=envelope_results,
            fft_ratios=fft_ratios,
            quality_score=file_quality
        )
        
        final_fault_label = fusion_result['fault_label']
        confidence = fusion_result['confidence']
        
    print(f"   ✓ Fault classification: {final_fault_label}")
    print(f"   ✓ Confidence: {confidence:.3f}")
    
    # ========================================
    # Step 6: Health Classification
    # ========================================
    print("\n💚 Step 6: Health classification...")
    
    # Map fault label to health status
    if final_fault_label == 'Normal' and confidence >= 0.6:
        health_status = 'Healthy'
    else:
        health_status = 'Unhealthy'
    
    print(f"   ✓ Health status: {health_status}")
    
    # ========================================
    # Step 7: Generate Explanation
    # ========================================
    print("\n📋 Step 7: Generating explanation...")
    
    rationale = generate_diagnosis_rationale(
        td_indicators=td_indicators,
        envelope_results=envelope_results,
        fault_label=final_fault_label,
        confidence=confidence,
        quality_score=file_quality
    )
    
    print("   ✓ Key evidence:")
    for key, value in rationale['key_evidence'].items():
        print(f"     - {key}: {value}")
    
    # ========================================
    # Final Results
    # ========================================
    print("\n" + "="*60)
    print("🎯 FINAL DIAGNOSIS RESULTS")
    print("="*60)
    
    results = {
        'file_path': file_path,
        'health': health_status,
        'fault_label': final_fault_label,
        'confidence': confidence,
        'processing_metadata': {
            'sensor': signal_data['sensor'],
            'load': load,
            'rpm': rpm,
            'sampling_frequency': signal_data['fs'],
            'algorithm_version': 'BDX-2024.1'
        },
        'quality': {
            'file_quality': file_quality,
            'valid_segments': len(valid_segments),
            'total_segments': len(segments)
        },
        'td_evidence': {
            'mean_kurtosis': td_indicators.mean_kurtosis,
            'kurtosis_burst_index': td_indicators.kurtosis_burst_index,
            'rms_trend_slope': td_indicators.rms_trend_slope,
            'mean_crest_factor': td_indicators.mean_crest_factor
        },
        'fd_evidence': {
            'selected_envelope_band': envelope_results.selected_band,
            'fault_frequency_snrs': envelope_results.fault_snrs,
            'envelope_energy': envelope_results.envelope_energy
        },
        'rationale': rationale
    }
    
    print(f"Health Status: {health_status}")
    print(f"Fault Type: {final_fault_label}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Signal Quality: {file_quality:.1%}")
    
    return results

# ========================================
# Example Helper Functions
# ========================================

def load_mat_file(file_path: str, sensor: str = 'DE') -> Dict[str, Any]:
    """Simulate loading .mat file (would use scipy.io.loadmat)."""
    import scipy.io
    
    mat_data = scipy.io.loadmat(file_path)
    
    # Find sensor data (CWRU convention)
    sensor_key = None
    for key in mat_data.keys():
        if key.endswith(f'_{sensor}_time'):
            sensor_key = key
            break
            
    if sensor_key is None:
        raise ValueError(f"Sensor {sensor} not found in file")
        
    signal = mat_data[sensor_key].flatten()
    
    # Auto-detect sampling frequency (typically 12 kHz for CWRU)
    fs = 12000.0
    
    return {
        'signal': signal,
        'fs': fs,
        'sensor': sensor,
        'available_sensors': ['DE', 'FE'],  # Would be detected from file
        'metadata': {'filename': Path(file_path).name}
    }

def infer_load_from_filename(filename: str) -> Optional[int]:
    """Infer load from CWRU filename patterns."""
    if any(x in filename for x in ['097', '098', '099']):
        return 0  # 0 HP
    elif any(x in filename for x in ['105', '106', '107']):
        return 1  # 1 HP
    elif any(x in filename for x in ['118', '119', '120']):
        return 2  # 2 HP
    elif any(x in filename for x in ['130', '131', '132']):
        return 3  # 3 HP
    return None

def get_rpm_from_load(load: int) -> float:
    """Map load to RPM."""
    rpm_map = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
    return rpm_map.get(load, 1797)

# Additional helper functions would be implemented here...
# (segment_signal, compute_td_features, compute_envelope_analysis, etc.)

# ========================================
# Example Usage
# ========================================

if __name__ == '__main__':
    # Example configuration
    config = ProcessingConfig(
        window_size=2048,
        overlap=0.5,
        normalization='zscore',
        min_file_quality=0.6,
        sensor_preference='DE'
    )
    
    # Example file path (would be actual CWRU .mat file)
    example_file = "data/normal_0_97.mat"
    
    try:
        results = end_to_end_bearing_diagnosis(example_file, config)
        
        # Save results
        import json
        with open('diagnosis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to: diagnosis_results.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
```

## Expected Output Example

```
🔍 Starting BearingDX analysis of: normal_0_97.mat
============================================================
📁 Step 1: Loading .mat file...
   ✓ Signal loaded: 121344 samples @ 12000.0 Hz
   ✓ Available sensors: ['DE', 'FE']
   ✓ Primary sensor: DE
   ✓ Detected load: 0 HP, RPM: 1797

🔧 Step 2: Signal segmentation and quality assessment...
   ✓ Total segments: 118
   ✓ Valid segments: 112 (94.9%)
   ✓ File quality score: 0.949

📊 Step 3: Time domain feature extraction...
   ✓ Mean kurtosis: 0.12
   ✓ Max kurtosis: 1.85
   ✓ Kurtosis burst index: 5.4%
   ✓ Mean crest factor: 4.23
   ✓ RMS trend slope: -0.000012
   ✓ TD fault suggestion: Normal (confidence: 0.78)

🌊 Step 4: Frequency domain analysis...
   ✓ Fault frequencies @ 1797 RPM:
     - BPFI: 162.2 Hz
     - BPFO: 107.4 Hz
     - BSF: 141.2 Hz
     - FTF: 23.7 Hz
   ✓ Using top 28 segments for FD analysis
   ✓ Selected envelope band: 2000-5000 Hz
   ✓ Envelope SNRs:
     - BPFI: 1.12
     - BPFO: 1.08
     - BSF: 1.15
     - FTF: 1.31

🔀 Step 5: Evidence fusion and fault classification...
   ✓ Passed Normal fast-path gate
   ✓ Fault classification: Normal
   ✓ Confidence: 0.850

💚 Step 6: Health classification...
   ✓ Health status: Healthy

📋 Step 7: Generating explanation...
   ✓ Key evidence:
     - Low TD indicators: All within normal ranges
     - Quiet envelope spectrum: No significant fault frequency peaks
     - Good signal quality: 94.9% valid segments
     - Consistent crest factors: Low variability across segments

============================================================
🎯 FINAL DIAGNOSIS RESULTS
============================================================
Health Status: Healthy
Fault Type: Normal
Confidence: 85.0%
Signal Quality: 94.9%

💾 Results saved to: diagnosis_results.json
```

## CLI Usage Examples

### Single File Analysis
```bash
# Basic classification
python -m bearingdx.cli.dx classify data/bearing_001.mat --sensor DE --output results.json

# With load specification and debug info
python -m bearingdx.cli.dx classify data/bearing_001.mat --sensor DE --load 0 --debug --output results.json

# Using conservative profile
python -m bearingdx.cli.dx classify data/bearing_001.mat --config conservative --output results.json
```

### Batch Processing
```bash
# Process all .mat files in directory
python -m bearingdx.cli.dx batch data/ --output batch_results.csv

# Recursive search with pattern matching
python -m bearingdx.cli.dx batch data/ --pattern "*_DE_*.mat" --recursive --output results.csv

# Using sensitive configuration profile
python -m bearingdx.cli.dx batch data/ --config sensitive --output sensitive_results.csv
```

### Signal Inspection
```bash
# Parse and inspect file
python -m bearingdx.cli.dx parse data/bearing_001.mat --show-segments --show-quality

# Debug mode with full details
python -m bearingdx.cli.dx parse data/bearing_001.mat --debug --show-features
```

### Quality Assessment
```bash
# Build quality baseline from Normal files
python -m bearingdx.cli.dx quality data/normal/ --build-baseline --output normal_baseline.json

# Assess quality using existing baseline
python -m bearingdx.cli.dx quality data/test_bearing.mat --baseline normal_baseline.json
```

### Configuration Management
```bash
# List available profiles
python -m bearingdx.cli.dx config --list-profiles

# Show configuration values
python -m bearingdx.cli.dx config --profile conservative --show

# Validate configuration
python -m bearingdx.cli.dx config --profile sensitive --validate
```

This example demonstrates the complete end-to-end workflow with realistic processing steps, output formatting, and CLI integration that would be implemented in the full BearingDX system.