# BearingDX Core Interfaces and Data Models

## Data Models and Type Definitions

### Core Data Structures

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

@dataclass
class BearingGeometry:
    """Bearing geometric parameters for fault frequency calculation."""
    rolling_elements: int = 9  # Number of rolling elements
    ball_diameter_mm: float = 7.938  # Ball diameter in mm
    pitch_diameter_mm: float = 39.04  # Pitch diameter in mm
    contact_angle_deg: float = 0.0  # Contact angle in degrees
    
@dataclass
class FaultFrequencies:
    """Calculated fault frequencies for a given RPM."""
    rpm: float
    bpfi: float  # Ball Pass Frequency Inner race
    bpfo: float  # Ball Pass Frequency Outer race
    bsf: float   # Ball Spin Frequency
    ftf: float   # Fundamental Train Frequency
    
@dataclass
class SignalData:
    """Raw signal data from .mat file."""
    signal: np.ndarray
    fs: float  # Sampling frequency
    sensor: str  # Primary sensor ('DE', 'FE', 'BA')
    available_sensors: List[str]
    metadata: Dict[str, Any]
    filename: str
    
@dataclass
class SegmentData:
    """Individual signal segment with metadata."""
    data: np.ndarray
    index: int  # Segment index in file
    start_sample: int
    end_sample: int
    is_valid: bool
    quality_score: float
    
@dataclass
class TDFeatures:
    """Time-domain features for a single segment."""
    # Basic statistical features
    mean: float
    std: float
    rms: float
    peak: float
    peak_to_peak: float
    variance: float
    
    # Shape and distribution features
    skewness: float
    kurtosis: float
    
    # Bearing-specific features
    crest_factor: float  # peak / rms
    form_factor: float   # rms / mean_abs
    impulse_factor: float  # peak / mean_abs
    margin_factor: float   # peak / mean_root_square
    shape_factor: float    # rms / mean_abs

@dataclass
class TDIndicators:
    """File-level time-domain indicators."""
    # Degradation indicators
    kurtosis_burst_index: float  # % segments with kurtosis > 5
    rms_trend_slope: float  # Linear trend in RMS
    peak_rms_ratio_sd: float  # Stability of peak/RMS ratio
    degradation_acceleration: float  # Quadratic term in RMS trend
    transient_event_density: float  # % segments with high peaks
    crest_factor_stability: float  # SD of crest factor
    
    # Overall statistics
    mean_kurtosis: float
    max_kurtosis: float
    mean_crest_factor: float
    rms_variability: float
    
@dataclass
class QualityScore:
    """Quality assessment for a segment or file."""
    score: float  # Overall quality [0,1]
    is_valid: bool
    warnings: List[str]
    flags: Dict[str, bool]  # clipping, low_variance, dc_drift, etc.
    
@dataclass
class QualityReport:
    """File-level quality assessment."""
    file_quality: float  # % valid segments
    valid_segments: int
    total_segments: int
    segment_scores: List[QualityScore]
    overall_warnings: List[str]
    
@dataclass
class EnvelopeResults:
    """Results from envelope analysis."""
    # Selected envelope band
    selected_band: Tuple[float, float]
    band_selection_reason: str
    
    # Fault frequency analysis
    fault_snrs: Dict[str, float]  # BPFI, BPFO, BSF, FTF SNRs
    sideband_ratios: Dict[str, float]  # Sideband energy ratios
    harmonic_content: Dict[str, List[float]]  # Harmonic peaks
    
    # Overall envelope metrics
    envelope_energy: float
    cyclic_content: float
    noise_floor: float
    
@dataclass
class FFTRatios:
    """FFT-based energy ratios around fault frequencies."""
    fault_energy_ratios: Dict[str, float]  # Energy near fault frequencies
    broadband_energy: float
    frequency_domain_snr: float
    spectral_kurtosis: float
    
@dataclass
class DiagnosisResult:
    """Final diagnosis result from fusion engine."""
    # Primary outputs
    health: str  # 'Healthy' or 'Unhealthy'
    fault_label: str  # 'Normal', 'IR', 'OR', 'Ball', 'Uncertain'
    confidence: float  # [0, 1]
    
    # Explanation and rationale
    rationale: Dict[str, Any]  # Key evidence and reasoning
    td_evidence: Dict[str, float]  # Top TD indicators
    fd_evidence: Dict[str, float]  # Top FD indicators
    quality_evidence: Dict[str, Any]  # Quality metrics
    
    # Processing metadata
    sensors_used: List[str]
    processing_time: float
    algorithm_version: str
```

## Module Interfaces

### bdxio.loader Module

```python
def load_mat(file_path: str, 
             sensor: str = 'DE',
             auto_detect: bool = True,
             fs_override: Optional[float] = None) -> SignalData:
    """
    Load CWRU .mat file with enhanced metadata extraction.
    
    Args:
        file_path: Path to .mat file
        sensor: Preferred sensor ('DE', 'FE', 'BA')
        auto_detect: Auto-detect available sensors and sampling rate
        fs_override: Override sampling frequency if needed
        
    Returns:
        SignalData with signal, metadata, and sensor information
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no valid sensors found or invalid format
        RuntimeError: If .mat file is corrupted or unreadable
    """

def detect_available_sensors(mat_data: Dict[str, Any]) -> List[str]:
    """
    Detect available sensors in .mat file based on variable naming.
    
    Args:
        mat_data: Loaded .mat file data dictionary
        
    Returns:
        List of available sensor names ('DE', 'FE', 'BA')
    """

def infer_sampling_rate(mat_data: Dict[str, Any], 
                       signal_name: str) -> Optional[float]:
    """
    Infer sampling rate from .mat file metadata or signal properties.
    
    Args:
        mat_data: Loaded .mat file data dictionary
        signal_name: Name of the signal variable
        
    Returns:
        Sampling rate in Hz, or None if cannot be determined
    """
```

### bdxio.segmenter Module

```python
def segment_signal(signal: np.ndarray,
                  fs: float,
                  window_size: int = 2048,
                  overlap: float = 0.5,
                  normalization: str = 'zscore') -> List[SegmentData]:
    """
    Segment signal with optimized processing and quality assessment.
    
    Args:
        signal: Input time series signal
        fs: Sampling frequency in Hz
        window_size: Segment length in samples
        overlap: Overlap fraction [0, 1)
        normalization: 'zscore', 'rms', 'minmax', or 'none'
        
    Returns:
        List of SegmentData objects with quality scores
        
    Performance Notes:
        - Single-pass processing with minimal memory copies
        - Vectorized normalization operations
        - Early quality filtering to reduce downstream computation
    """

def normalize_segment(segment: np.ndarray, 
                     method: str = 'zscore') -> np.ndarray:
    """
    Normalize segment using specified method.
    
    Args:
        segment: Raw signal segment
        method: Normalization method ('zscore', 'rms', 'minmax')
        
    Returns:
        Normalized segment
    """

def assess_segment_quality(segment: np.ndarray,
                          quality_config: Dict[str, float]) -> QualityScore:
    """
    Assess quality of individual segment.
    
    Args:
        segment: Signal segment to assess
        quality_config: Quality thresholds and parameters
        
    Returns:
        QualityScore with validity flag and warnings
        
    Quality Checks:
        - Clipping detection (samples at ±full scale)
        - Low variance detection (insufficient signal content)
        - DC drift detection (excessive mean offset)
        - Length validation (minimum required samples)
    """
```

### bdxio.cwru Module

```python
def get_fault_frequencies(rpm: float, 
                         geometry: BearingGeometry) -> FaultFrequencies:
    """
    Calculate bearing fault frequencies for given RPM and geometry.
    
    Args:
        rpm: Shaft rotation speed in RPM
        geometry: Bearing geometric parameters
        
    Returns:
        FaultFrequencies object with BPFI, BPFO, BSF, FTF
        
    Formulas:
        - BPFI = (Z/2) * (1 + (Bd/Pd) * cos(θ)) * (RPM/60)
        - BPFO = (Z/2) * (1 - (Bd/Pd) * cos(θ)) * (RPM/60) 
        - BSF = (Pd/2Bd) * (1 - (Bd/Pd)² * cos²(θ)) * (RPM/60)
        - FTF = (1/2) * (1 - (Bd/Pd) * cos(θ)) * (RPM/60)
    """

def infer_load_from_filename(filename: str) -> Optional[int]:
    """
    Extract load information from CWRU filename patterns.
    
    Args:
        filename: CWRU .mat filename
        
    Returns:
        Load value (0-3) or None if cannot be determined
        
    Patterns Recognized:
        - X097, X098, X099 → Load 0
        - X105, X106, X107 → Load 1  
        - X118, X119, X120 → Load 2
        - X130, X131, X132 → Load 3
    """

def get_rpm_from_load(load: int, 
                     rpm_map: Dict[int, float] = None) -> float:
    """
    Map load value to RPM using configurable mapping.
    
    Args:
        load: Load value (0-3)
        rpm_map: Custom load→RPM mapping, uses default if None
        
    Returns:
        RPM value for the specified load
        
    Default Mapping:
        - 0 HP → 1797 RPM
        - 1 HP → 1772 RPM  
        - 2 HP → 1750 RPM
        - 3 HP → 1730 RPM
    """

def parse_cwru_metadata(filename: str) -> Dict[str, Any]:
    """
    Parse CWRU filename to extract fault type and load information.
    
    Args:
        filename: CWRU .mat filename
        
    Returns:
        Dictionary with fault_type, load, sensor, and other metadata
    """
```

### features.td Module (Enhanced)

```python
def compute_td_features(segment: np.ndarray) -> TDFeatures:
    """
    Compute comprehensive time-domain features for a segment.
    
    Args:
        segment: Normalized signal segment
        
    Returns:
        TDFeatures object with all computed features
        
    Features Computed:
        - Statistical: mean, std, rms, peak, variance, skewness, kurtosis
        - Shape factors: crest, form, impulse, margin, shape factors
        - Bearing-specific indicators for fault detection
        
    Performance:
        - Vectorized operations for speed
        - Robust handling of edge cases (zero variance, etc.)
    """

def compute_file_td_indicators(segments_features: List[TDFeatures],
                              segment_quality: List[QualityScore]) -> TDIndicators:
    """
    Compute file-level TD indicators from segment features.
    
    Args:
        segments_features: TD features for all segments
        segment_quality: Quality scores for filtering
        
    Returns:
        TDIndicators with file-level degradation measures
        
    Indicators Computed:
        - Kurtosis Burst Index: % segments with kurtosis > threshold
        - RMS Trend Slope: Linear regression slope of RMS over time
        - Peak-RMS Ratio SD: Stability measure of peak/RMS ratio
        - Degradation Acceleration: Quadratic component in RMS trend
        - Transient Event Density: % segments with exceptional peaks
        - Crest Factor Stability: Consistency of crest factor values
    """

def infer_fault_location_td(td_indicators: TDIndicators,
                           quality_report: QualityReport,
                           thresholds: Dict[str, float]) -> Tuple[str, float]:
    """
    Infer fault type from TD indicators using rule-based logic.
    
    Args:
        td_indicators: File-level TD indicators
        quality_report: Signal quality assessment
        thresholds: Classification thresholds
        
    Returns:
        Tuple of (fault_type, confidence) where fault_type is one of:
        'Normal', 'IR', 'OR', 'Ball', 'Uncertain'
        
    Logic:
        - High kurtosis + burst patterns → Inner/Outer race faults
        - Distributed high kurtosis → Ball fault
        - Low overall indicators + good quality → Normal
        - Poor quality or conflicting evidence → Uncertain
    """
```

### features.fd Module (Enhanced)

```python
def compute_envelope_analysis(signal: np.ndarray,
                            fs: float,
                            rpm: float,
                            geometry: BearingGeometry,
                            candidate_bands: List[Tuple[float, float]] = None,
                            auto_select_band: bool = True) -> EnvelopeResults:
    """
    Perform comprehensive envelope analysis with automatic band selection.
    
    Args:
        signal: Time domain signal (single segment or concatenated)
        fs: Sampling frequency
        rpm: Shaft speed for fault frequency calculation
        geometry: Bearing geometry parameters
        candidate_bands: List of frequency bands to test
        auto_select_band: Enable automatic band selection
        
    Returns:
        EnvelopeResults with fault frequency analysis and band selection
        
    Process:
        1. Test each candidate envelope band
        2. Select band with highest fault frequency SNR
        3. Compute envelope spectrum using Hilbert transform
        4. Calculate SNRs at fault frequencies and harmonics
        5. Analyze sideband content around fault frequencies
        6. Assess cyclic content vs noise floor
    """

def compute_fft_ratios(signal: np.ndarray,
                      fs: float,
                      fault_freqs: FaultFrequencies,
                      frequency_resolution: float = 1.0) -> FFTRatios:
    """
    Compute FFT-based energy ratios around fault frequencies.
    
    Args:
        signal: Time domain signal
        fs: Sampling frequency  
        fault_freqs: Target fault frequencies
        frequency_resolution: Frequency bin width for energy integration
        
    Returns:
        FFTRatios with energy measurements around fault frequencies
        
    Metrics:
        - Energy ratios within ±resolution of each fault frequency
        - Broadband energy level for normalization
        - Spectral kurtosis for transient detection
        - Overall frequency domain SNR
    """

def select_optimal_envelope_band(signal: np.ndarray,
                               fs: float,
                               fault_freqs: FaultFrequencies,
                               candidate_bands: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], str]:
    """
    Select optimal envelope band based on fault frequency content.
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        fault_freqs: Target fault frequencies
        candidate_bands: Bands to evaluate
        
    Returns:
        Tuple of (selected_band, selection_reason)
        
    Selection Criteria:
        - Maximize sum of fault frequency SNRs
        - Prefer bands with clear cyclic content
        - Avoid bands dominated by noise or broadband energy
    """

def prewhiten_signal(signal: np.ndarray,
                    fs: float,
                    smooth_bins: int = 41) -> np.ndarray:
    """
    Pre-whiten signal to enhance transient components.
    
    Args:
        signal: Input time series
        fs: Sampling frequency
        smooth_bins: Smoothing kernel size for spectral flattening
        
    Returns:
        Pre-whitened signal with enhanced impulsive content
        
    Process:
        - Compute magnitude spectrum
        - Smooth spectrum to estimate background
        - Flatten spectrum by dividing by smoothed version
        - Transform back to time domain
    """
```

### features.quality Module

```python
def build_quality_baselines(normal_files: List[str],
                          sensor: str = 'DE',
                          config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Build quality baselines from Normal condition files.
    
    Args:
        normal_files: List of paths to Normal condition .mat files
        sensor: Target sensor for baseline construction
        config: Processing configuration
        
    Returns:
        Dictionary with p5-p95 ranges for key TD features
        
    Process:
        - Load and process all Normal files
        - Extract TD features from all segments
        - Compute percentile ranges (p5, p50, p95) for each feature
        - Store as quality reference for outlier detection
    """

def assess_signal_quality(segments: List[SegmentData],
                         baselines: Optional[Dict[str, Any]] = None) -> QualityReport:
    """
    Assess overall signal quality across all segments.
    
    Args:
        segments: List of signal segments with individual quality scores
        baselines: Optional quality baselines for comparison
        
    Returns:
        QualityReport with file-level quality assessment
        
    Assessment:
        - Calculate % of valid segments
        - Identify common quality issues
        - Flag files with insufficient valid data
        - Provide recommendations for processing
    """

def detect_signal_anomalies(segment: np.ndarray,
                           fs: float,
                           thresholds: Dict[str, float]) -> List[str]:
    """
    Detect various signal quality anomalies.
    
    Args:
        segment: Signal segment to analyze
        fs: Sampling frequency
        thresholds: Detection thresholds
        
    Returns:
        List of detected anomaly types
        
    Anomaly Types:
        - 'clipping': Signal saturated at limits
        - 'low_variance': Insufficient signal content  
        - 'dc_drift': Excessive DC offset
        - 'impulse_noise': Non-bearing related impulses
        - 'too_short': Insufficient segment length
    """
```

### engine.fusion Module (Enhanced)

```python
def diagnose_bearing_fault(td_indicators: TDIndicators,
                          fd_results: EnvelopeResults,
                          quality_report: QualityReport,
                          config: Dict[str, Any]) -> DiagnosisResult:
    """
    Main fusion engine implementing TD screening + FD confirmation.
    
    Args:
        td_indicators: File-level time-domain indicators
        fd_results: Envelope analysis results
        quality_report: Signal quality assessment
        config: Fusion configuration and thresholds
        
    Returns:
        DiagnosisResult with final classification and evidence
        
    Workflow:
        1. Quality Gate: Ensure ≥60% valid segments
        2. TD Screening: Initial fault type suggestion
        3. Normal Fast Path: Envelope-only Normal gate  
        4. FD Confirmation: Validate TD suggestion with envelope evidence
        5. Ball Fault Logic: Special handling for ball faults
        6. Confidence Scoring: Evidence strength assessment
        7. Multi-sensor Fusion: Combine results if multiple sensors
    """

def apply_normal_fast_path(td_indicators: TDIndicators,
                          envelope_results: EnvelopeResults,
                          thresholds: Dict[str, float]) -> Optional[str]:
    """
    Fast path for Normal classification using envelope-only evidence.
    
    Args:
        td_indicators: TD analysis results
        envelope_results: Envelope analysis results
        thresholds: Normal classification thresholds
        
    Returns:
        'Normal' if fast path succeeds, None otherwise
        
    Logic:
        - Requires quiet TD indicators AND low envelope energy
        - Uses envelope-only to avoid FFT broadband masking
        - Conservative thresholds to minimize false negatives
    """

def fuse_multisensor_results(results: Dict[str, DiagnosisResult],
                           sensor_weights: Dict[str, float],
                           quality_scores: Dict[str, float]) -> DiagnosisResult:
    """
    Fuse results from multiple sensors (DE, FE, BA).
    
    Args:
        results: Per-sensor diagnosis results  
        sensor_weights: Relative sensor importance weights
        quality_scores: Per-sensor quality scores
        
    Returns:
        Fused DiagnosisResult combining all sensor evidence
        
    Strategy:
        - Weight results by sensor importance and quality
        - Handle disagreements with Uncertain classification
        - Require consensus for high-confidence results
        - Fall back to best single sensor if fusion unclear
    """

def compute_diagnosis_confidence(td_evidence: Dict[str, float],
                               fd_evidence: Dict[str, float],
                               fault_label: str,
                               quality_score: float) -> float:
    """
    Compute diagnosis confidence based on evidence strength.
    
    Args:
        td_evidence: TD indicator scores
        fd_evidence: FD indicator scores  
        fault_label: Diagnosed fault type
        quality_score: Overall signal quality
        
    Returns:
        Confidence score [0, 1]
        
    Factors:
        - Evidence strength and consistency
        - Signal quality impact
        - Fault type specific reliability
        - Multi-evidence agreement
    """
```

### engine.classify Module

```python
def classify_bearing_health(diagnosis: DiagnosisResult,
                          config: Dict[str, Any]) -> str:
    """
    Map fault diagnosis to binary health classification.
    
    Args:
        diagnosis: Diagnosis result from fusion engine
        config: Classification configuration
        
    Returns:
        'Healthy' or 'Unhealthy'
        
    Rules:
        - Healthy: fault_label == 'Normal' AND confidence ≥ threshold
        - Unhealthy: All other cases (IR/OR/Ball/Uncertain)
        - Configurable for conservative vs sensitive modes
    """

def apply_classification_rules(fault_label: str,
                             confidence: float,
                             quality_score: float,
                             rules: Dict[str, Any]) -> str:
    """
    Apply configurable classification rules.
    
    Args:
        fault_label: Diagnosed fault type
        confidence: Diagnosis confidence
        quality_score: Signal quality score
        rules: Classification rule configuration
        
    Returns:
        Health classification ('Healthy' or 'Unhealthy')
    """
```

### engine.cache Module

```python
class AnalysisCache:
    """Performance cache for batch processing optimization."""
    
    def __init__(self, cache_size: int = 100):
        """Initialize cache with specified size limit."""
        
    def get_fft_window(self, window_size: int) -> np.ndarray:
        """Get cached FFT window function."""
        
    def get_fault_frequencies(self, rpm: float, 
                            geometry: BearingGeometry) -> FaultFrequencies:
        """Get cached fault frequencies for RPM/geometry combo."""
        
    def get_envelope_spectrum(self, signal_hash: str,
                            band: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get cached envelope spectrum if available."""
        
    def cache_envelope_spectrum(self, signal_hash: str,
                              band: Tuple[float, float],
                              spectrum: np.ndarray) -> None:
        """Cache computed envelope spectrum."""
        
    def clear(self) -> None:
        """Clear all cached data."""
```

## Configuration Interface

```python
@dataclass
class BearingDXConfig:
    """Main configuration object for BearingDX system."""
    
    # Signal processing
    segmentation: Dict[str, Any]
    normalization: str = 'zscore'
    
    # Bearing parameters
    bearing_geometry: BearingGeometry
    load_rpm_map: Dict[int, float]
    
    # Feature extraction
    envelope_bands: List[Tuple[float, float]]
    auto_select_band: bool = True
    
    # Quality assessment  
    quality_thresholds: Dict[str, float]
    min_file_quality: float = 0.6
    
    # Fusion and classification
    fusion_thresholds: Dict[str, float]
    sensor_weights: Dict[str, float]
    classification_mode: str = 'conservative'  # or 'sensitive'
    
    # Performance
    cache_enabled: bool = True
    max_cache_size: int = 100
    use_top_segments_only: bool = True
    top_segment_fraction: float = 0.25

def load_config(config_path: str, 
               profile: str = 'default') -> BearingDXConfig:
    """
    Load configuration from YAML files with profile inheritance.
    
    Args:
        config_path: Path to configuration directory
        profile: Configuration profile ('default', 'conservative', 'sensitive')
        
    Returns:
        BearingDXConfig object with loaded settings
    """

def validate_config(config: BearingDXConfig) -> List[str]:
    """
    Validate configuration parameters and return any errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
```

## Error Handling and Logging

```python
class BearingDXError(Exception):
    """Base exception for BearingDX-specific errors."""
    
class FileProcessingError(BearingDXError):
    """Error in file loading or processing."""
    
class FeatureExtractionError(BearingDXError):
    """Error in feature computation."""
    
class DiagnosisError(BearingDXError):
    """Error in fault diagnosis process."""

def setup_logging(level: str = 'INFO', 
                 log_file: Optional[str] = None) -> None:
    """
    Configure logging for BearingDX system.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """
```

These interfaces provide a comprehensive foundation for implementing the enhanced BearingDX system while maintaining clean separation of concerns and enabling future extensibility.