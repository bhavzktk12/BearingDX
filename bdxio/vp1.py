"""
BearingDX - Phase-1 Vibration Parser
- Baseline-aware quality (bk1)
- Time-domain fault inference (dx1)
- Load→RPM mapping in metadata
"""

import os, json
import numpy as np
import scipy.io
from datetime import datetime
from typing import List, Dict, Any

from features.td import compute_time_domain_features, compute_diagnostic_indicators
from dx1 import summarize_segment_features, predict_fault_location_ml
from bk1 import validate_signal_quality, bearing_knowledge, RPM_BY_LOAD


DEFAULT_SAMPLING_RATE = 12000  # Hz
SEGMENT_SIZE = 2048
OVERLAP_RATIO = 0.5

class VibrationSignalProcessor:
    def __init__(self, segment_size: int = SEGMENT_SIZE, overlap: float = OVERLAP_RATIO):
        self.segment_size = segment_size
        self.overlap = overlap
        self.hop_size = int(segment_size * (1 - overlap))

    def segment_signal(self, signal: np.ndarray, sampling_rate: int) -> List[Dict[str, Any]]:
        if len(signal) < self.segment_size:
            return [{
                'segment_id': 0, 'signal': signal, 'start_sample': 0,
                'end_sample': len(signal), 'duration_sec': len(signal)/sampling_rate,
                'is_full_window': False
            }]
        segments=[]
        for i, start in enumerate(range(0, len(signal)-self.segment_size+1, self.hop_size)):
            end = start + self.segment_size
            segments.append({'segment_id': i, 'signal': signal[start:end],
                             'start_sample': start, 'end_sample': end,
                             'duration_sec': self.segment_size/sampling_rate,
                             'is_full_window': True})
        return segments

    def normalize_signal(self, signal: np.ndarray, method: str = 'zscore') -> np.ndarray:
        if method == 'zscore':
            mean = np.mean(signal); std = np.std(signal)
            return (signal - mean) / std if std > 0 else signal - mean
        elif method == 'minmax':
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) if np.max(signal) > np.min(signal) else signal
        elif method == 'rms':
            rms = np.sqrt(np.mean(signal**2)); return signal / rms if rms > 0 else signal
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

def parse_vibration_file(file_path: str, sensor_key: str = 'DE', sampling_rate: int = DEFAULT_SAMPLING_RATE) -> Dict[str, Any]:
    result = {'success': False, 'filename': os.path.basename(file_path), 'segments': [],
              'metadata': {}, 'warnings': [], 'errors': [], 'processing_log': []}
    try:
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")

        mat = scipy.io.loadmat(file_path)
        variable_key = next((k for k in mat if k.endswith(f"{sensor_key}_time")), None)
        if variable_key is None: raise ValueError(f"No sensor data found for key '*_{sensor_key}_time'")

        signal = mat[variable_key].squeeze()
        if signal.ndim > 1: signal = signal.flatten()

        processor = VibrationSignalProcessor()
        segments = processor.segment_signal(signal, sampling_rate)

        processed=[]
        for seg in segments:
            feats = compute_time_domain_features(seg['signal'])
            q = validate_signal_quality(feats, sensor_key=sensor_key)
            processed.append({
                "segment_id": seg['segment_id'], "filename": result['filename'],
                "sampling_rate": sampling_rate, "duration_sec": seg['duration_sec'],
                "start_sample": seg['start_sample'], "end_sample": seg['end_sample'],
                "is_full_window": seg['is_full_window'], **feats,
                "quality_score": q['quality_score'], "is_valid": q['is_valid'],
                "quality_warnings": q['warnings'],
                "normalized_signal": processor.normalize_signal(seg['signal'], 'zscore').tolist(),
                "timestamp": datetime.now().isoformat()
            })

        result['segments'] = processed; result['success'] = True
        result['metadata'] = {
            "sampling_rate": sampling_rate,
            "segment_count": len(processed),
            "sensor_key": sensor_key,
            "signal_length": len(signal)
        }
        bearing_knowledge.learn_from_success(result['filename'], {"quality_score": float(np.mean([s["quality_score"] for s in processed]))})

    except Exception as e:
        msg = str(e); result['errors'].append(msg); result['processing_log'].append(f"Error: {msg}")
        bearing_knowledge.learn_from_failure(result['filename'], msg)

    return result

if __name__ == "__main__":
    import sys, re
    from bk1 import RPM_BY_LOAD

    def infer_load_from_name(path: str) -> int:
        m = re.search(r"(\d{3,5})(?=\.mat$)", os.path.basename(path))
        if m:
            rpm = int(m.group(1))
            # pick closest
            best = min(RPM_BY_LOAD.items(), key=lambda kv: abs(kv[1]-rpm))[0]
            return best
        m2 = re.search(r"normal_(\d)_\d+", os.path.basename(path), re.I)
        return int(m2.group(1)) if m2 else 1

    # Usage: py vp1.py [file_path] [load_hp] [sensor]
    test_file = (len(sys.argv) >= 2 and sys.argv[1]) or r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\Ball\B007_2_1750.mat"
    sensor    = (len(sys.argv) >= 4 and sys.argv[3]) or "DE"
    load_hp   = int(sys.argv[2]) if len(sys.argv) >= 3 else infer_load_from_name(test_file)

    print("BearingDX Vibration Parser - Phase 1.1 Test")
    print("=" * 60)
    if os.path.exists(test_file):
        rpm = RPM_BY_LOAD.get(load_hp, 1772)
        print(f"Sensor={sensor}  Load={load_hp}HP → RPM={rpm}")
        result = parse_vibration_file(test_file, sensor_key=sensor, sampling_rate=DEFAULT_SAMPLING_RATE)

        if result['success']:
            print(f"✅ Parsed: {result['filename']}  Segments={result['metadata']['segment_count']}")
            summarize_segment_features(result['segments'], top_n=3)

            indicators = compute_diagnostic_indicators(result)
            print("\n🔍 File-Level Indicators (key):")
            print("=" * 60)
            for k in ("rms_trend_slope","kurtosis_burst_index","peak_to_rms_ratio_sd","top_3_severity"):
                print(f"{k:>28}: {indicators['tier1'][k]:.4f}")
            print(f"{'transient_event_density':>28}: {indicators['tier2']['transient_event_density']:.2f}")
            print(f"{'impulse_persistence':>28}: {indicators['tier2']['impulse_persistence']:.2f}")

            # === ML inference (time-domain only, existing model) ===
            pred = predict_fault_location_ml(result)
            if pred.get("available"):
                print("\n🤖 Time-Domain Fault Inference (ML):")
                print("=" * 60)
                print(f"Predicted: {pred['label']}  | Confidence: {pred['confidence']:.2%}")
                print("Class probabilities:")
                for c, p in pred["probs"].items():
                    print(f"  - {c:<6}: {p:.2%}")
            else:
                print(f"Inference unavailable: {pred.get('reason')}")
        else:
            print("Parse failed:", result['errors'])
    else:
        print(f"File not found: {test_file}")
