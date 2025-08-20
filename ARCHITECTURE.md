# BearingDX Architecture Proposal

## Overview

BearingDX is a vibration-based bearing diagnostics tool that processes CWRU .mat files to classify bearing health (Healthy/Unhealthy) and fault types (Normal/IR/OR/Ball/Uncertain). The architecture prioritizes accuracy, speed, and modularity for future enhancements.

## Enhanced Folder Structure

```
bearingdx/
├── README.md                    # Project overview and setup
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── pyproject.toml              # Modern Python packaging config
├── .gitignore                   # Git ignore patterns
│
├── bearingdx/                   # Main package directory
│   ├── __init__.py             # Package initialization and version
│   │
│   ├── bdxio/                   # Input/Output and data handling
│   │   ├── __init__.py
│   │   ├── loader.py           # Enhanced .mat reader with auto-detection
│   │   ├── segmenter.py        # Signal segmentation and normalization
│   │   └── cwru.py             # CWRU dataset utilities and metadata
│   │
│   ├── features/                # Feature extraction modules
│   │   ├── __init__.py
│   │   ├── td.py               # Time-domain features (enhanced)
│   │   ├── fd.py               # Frequency-domain features (enhanced)
│   │   └── quality.py          # Signal quality assessment
│   │
│   ├── engine/                  # Core processing and fusion
│   │   ├── __init__.py
│   │   ├── fusion.py           # Enhanced TD+FD fusion logic
│   │   ├── classify.py         # Health classification and thresholds
│   │   ├── vectorize.py        # Feature vectorization for ML
│   │   └── cache.py            # Performance caching system
│   │
│   ├── models/                  # Machine learning models (future)
│   │   ├── __init__.py
│   │   ├── baseline.py         # Baseline rule-based classifier
│   │   └── calibrated.py       # Calibrated ML models
│   │
│   ├── config/                  # Configuration management
│   │   ├── __init__.py
│   │   ├── defaults.yaml       # Default configuration
│   │   ├── conservative.yaml   # Conservative thresholds
│   │   ├── sensitive.yaml      # Sensitive thresholds
│   │   ├── bearing_geometry.yaml # Bearing specifications
│   │   └── loader.py           # Configuration loading utilities
│   │
│   ├── cli/                     # Command-line interface
│   │   ├── __init__.py
│   │   ├── dx.py               # Main CLI application
│   │   ├── batch.py            # Batch processing utilities
│   │   └── report.py           # Report generation
│   │
│   └── utils/                   # Shared utilities
│       ├── __init__.py
│       ├── logging.py          # Logging configuration
│       ├── validation.py       # Input validation
│       └── metrics.py          # Performance metrics
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py             # PyTest configuration
│   ├── fixtures/               # Test data and fixtures
│   │   ├── synthetic/          # Synthetic test signals
│   │   └── cwru_samples/       # Small CWRU test files
│   ├── unit/                   # Unit tests
│   │   ├── test_td_features.py
│   │   ├── test_fd_features.py
│   │   ├── test_fusion.py
│   │   ├── test_quality.py
│   │   └── test_loader.py
│   ├── integration/            # Integration tests
│   │   ├── test_end_to_end.py
│   │   └── test_batch_processing.py
│   └── golden/                 # Golden reference tests
│       ├── test_cwru_golden.py
│       └── golden_results.json
│
├── notebooks/                   # Analysis and development notebooks
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_validation.ipynb
│   └── 03_threshold_tuning.ipynb
│
├── scripts/                     # Utility scripts
│   ├── download_cwru.py        # CWRU dataset downloader
│   ├── calibrate_thresholds.py # Threshold calibration
│   └── benchmark.py            # Performance benchmarking
│
├── docs/                        # Documentation
│   ├── api.md                  # API documentation
│   ├── configuration.md        # Configuration guide
│   └── troubleshooting.md      # Common issues and solutions
│
└── reports/                     # Output directory (gitignored)
    ├── results/                # Analysis results
    ├── logs/                   # Application logs
    └── debug/                  # Debug artifacts
```

## Core Components and Responsibilities

### 1. bdxio Module - Data Input/Output

#### loader.py - Enhanced .mat File Reader
```python
def load_mat(file_path: str, sensor: str = 'DE', 
             auto_detect: bool = True) -> SignalData:
    """
    Load CWRU .mat file with automatic sensor detection and metadata extraction.
    
    Args:
        file_path: Path to .mat file
        sensor: Primary sensor ('DE', 'FE', 'BA')
        auto_detect: Auto-detect available sensors and sampling rate
        
    Returns:
        SignalData: Dataclass with signal, fs, sensors, metadata
    """
```

#### segmenter.py - Signal Processing
```python
def segment_signal(signal: np.ndarray, fs: float, 
                  win_size: int = 2048, overlap: float = 0.5,
                  normalize: str = 'zscore') -> List[SegmentData]:
    """
    Segment signal with configurable windowing and normalization.
    
    Performance optimized with single-pass processing and minimal copies.
    """

def assess_segment_quality(segment: np.ndarray, 
                         thresholds: QualityThresholds) -> QualityScore:
    """
    Assess individual segment quality for filtering.
    """
```

#### cwru.py - Dataset Utilities
```python
def get_fault_frequencies(rpm: float, bearing_geometry: BearingGeometry) -> FaultFreqs:
    """Calculate BPFI, BPFO, BSF, FTF for given RPM and bearing geometry."""

def infer_load_from_filename(filename: str) -> Optional[int]:
    """Extract load information from CWRU filename patterns."""

def get_rpm_from_load(load: int, rpm_map: Dict[int, float]) -> float:
    """Map load to RPM using configurable mapping."""
```

### 2. features Module - Feature Extraction

#### td.py - Time Domain Features (Enhanced)
```python
def compute_td_features(segment: np.ndarray) -> TDFeatures:
    """
    Compute comprehensive time-domain features per segment.
    
    Features: RMS, Peak, Crest Factor, Kurtosis, Skewness, Form Factor,
             Impulse Factor, Margin Factor, Shape Factor
    """

def compute_file_td_indicators(segments_features: List[TDFeatures]) -> TDIndicators:
    """
    Compute file-level TD indicators from segment features.
    
    Indicators:
    - Kurtosis Burst Index (% segments with kurtosis > 5)
    - RMS Trend Slope (linear regression over segments)
    - Peak-to-RMS Ratio SD (stability measure)
    - Degradation Acceleration (quadratic trend)
    - Transient Event Density (% high-peak segments)
    - Crest Factor Stability (consistency measure)
    """
```

#### fd.py - Frequency Domain Features (Enhanced)
```python
def compute_envelope_analysis(signal: np.ndarray, fs: float, rpm: float,
                            bands: List[Tuple[float, float]] = None,
                            bearing_geometry: BearingGeometry = None) -> EnvelopeResults:
    """
    Perform envelope analysis with automatic band selection.
    
    Process:
    1. Test candidate bands: [(1000,3000), (2000,5000), (3000,6000), (6000,10000)]
    2. Select band with highest fault frequency SNR
    3. Compute envelope spectrum and fault frequency peaks
    4. Calculate sidebands and harmonic content
    """

def compute_fft_ratios(signal: np.ndarray, fs: float, rpm: float,
                      bearing_geometry: BearingGeometry) -> FFTRatios:
    """
    Compute FFT-based energy ratios around fault frequencies.
    
    Returns energy ratios for BPFI, BPFO, BSF, FTF and their harmonics.
    """
```

#### quality.py - Signal Quality Assessment
```python
def build_quality_baselines(normal_files: List[str], 
                          sensor: str = 'DE') -> QualityBaselines:
    """
    Build per-sensor quality baselines from Normal condition files.
    
    Computes p5-p95 ranges for key TD features for quality gating.
    """

def assess_signal_quality(segments: List[SegmentData],
                        baselines: QualityBaselines) -> QualityReport:
    """
    Assess overall signal quality and flag issues.
    
    Checks:
    - Clipping detection (>0.5% samples at full scale)
    - Low variance detection (RMS < 1e-6g threshold)
    - DC drift detection (|mean| > 3×MAD)
    - Excessive noise detection
    - Segment length validation
    """
```

### 3. engine Module - Core Processing

#### fusion.py - Enhanced TD+FD Fusion
```python
def diagnose_bearing_fault(td_indicators: TDIndicators,
                          fd_results: EnvelopeResults,
                          quality_report: QualityReport,
                          config: FusionConfig) -> DiagnosisResult:
    """
    Main fusion logic implementing the TD screening → FD confirmation workflow.
    
    Process:
    1. TD Screening: Initial fault detection and type suggestion
    2. Quality Gate: Ensure sufficient data quality (≥60% valid segments)
    3. FD Confirmation: Envelope-based validation of fault type
    4. Multi-sensor Fusion: Combine DE/FE results with weighting
    5. Confidence Scoring: Compute final confidence based on evidence strength
    
    Returns:
        DiagnosisResult with health, fault_label, confidence, rationale
    """

def apply_normal_fast_path(td_indicators: TDIndicators,
                          envelope_results: EnvelopeResults,
                          thresholds: Dict[str, float]) -> Optional[str]:
    """
    Fast path for Normal classification using envelope-only checks.
    
    Avoids broadband FFT energy masking as fault indicators.
    """
```

#### classify.py - Health Classification
```python
def classify_bearing_health(diagnosis: DiagnosisResult,
                          config: ClassificationConfig) -> HealthResult:
    """
    Map fault diagnosis to binary health classification.
    
    Rules:
    - Healthy: fault_label == 'Normal' AND quality_score ≥ threshold
    - Unhealthy: All other cases (IR/OR/Ball/Uncertain)
    
    Configurable for conservative vs sensitive modes.
    """
```

#### cache.py - Performance Optimization
```python
class AnalysisCache:
    """
    Cache computed results for batch processing efficiency.
    
    Caches:
    - FFT windows and envelope spectra
    - Fault frequency templates per RPM
    - Quality baselines per sensor
    - Preprocessing results
    """
```

### 4. config Module - Configuration Management

#### Configuration Files

**defaults.yaml**
```yaml
# Signal Processing
segmentation:
  window_size: 2048
  overlap: 0.5
  normalization: 'zscore'

# Bearing Geometry (CWRU SKF 6205-2RS JEM)
bearing_geometry:
  rolling_elements: 9
  ball_diameter_mm: 7.938
  pitch_diameter_mm: 39.04
  contact_angle_deg: 0.0

# Load to RPM Mapping
load_rpm_map:
  0: 1797
  1: 1772
  2: 1750
  3: 1730

# Envelope Analysis
envelope:
  candidate_bands:
    - [1000, 3000]
    - [2000, 5000] 
    - [3000, 6000]
    - [6000, 10000]
  auto_select: true

# Quality Assessment
quality:
  min_file_quality: 0.6
  clipping_threshold: 0.005
  low_variance_threshold: 1e-6
  dc_drift_factor: 3.0

# Multi-sensor Fusion
sensor_weights:
  DE: 0.7
  FE: 0.3
  BA: 0.2
```

**conservative.yaml** - Inherit from defaults, override thresholds for fewer false positives
**sensitive.yaml** - Inherit from defaults, override thresholds for higher sensitivity

### 5. cli Module - Command Line Interface

#### dx.py - Main CLI
```python
# Usage Examples:
# Single file analysis
python -m bearingdx.cli.dx classify data/file.mat --sensor DE --output results.json

# Batch processing  
python -m bearingdx.cli.dx batch data/ --pattern "*.mat" --output results.csv

# Parse and inspect
python -m bearingdx.cli.dx parse data/file.mat --debug --show-segments

# Quality check
python -m bearingdx.cli.dx quality data/file.mat --baseline normal_baseline.json
```

## Key Enhancements Over Current Implementation

### 1. **Complete Configuration System**
- YAML-based configuration with inheritance
- Profile-based threshold management (conservative/sensitive)
- Runtime configuration validation

### 2. **Enhanced Quality Assessment**
- Systematic quality baseline construction from Normal files
- Per-segment and file-level quality scoring
- Quality-aware feature selection (top quartile segments for FD)

### 3. **Improved Multi-sensor Fusion**
- Configurable sensor weights (DE: 0.7, FE: 0.3)
- Quality-based sensor selection
- Disagreement detection and Uncertain classification

### 4. **Performance Optimizations**
- Analysis caching for batch processing
- Single-pass segmentation with minimal memory copies
- Quality-filtered FD analysis (only top segments)
- Vectorized feature computation

### 5. **Comprehensive Testing Framework**
- Unit tests for all feature computations
- Golden tests on CWRU reference files
- Integration tests for end-to-end workflows
- Synthetic signal generation for edge cases

### 6. **Production-Ready CLI**
- Batch processing with progress tracking
- Flexible output formats (JSON, CSV)
- Debug mode with intermediate results
- Error handling and validation

## End-to-End Processing Flow

```python
# Simplified end-to-end example
def process_bearing_file(file_path: str, config: Config) -> DiagnosisResult:
    # 1. Load and validate
    signal_data = load_mat(file_path, sensor=config.primary_sensor)
    
    # 2. Segment and quality check
    segments = segment_signal(signal_data.signal, signal_data.fs, 
                            config.segmentation)
    quality_report = assess_signal_quality(segments, config.quality_baselines)
    
    # 3. Extract features (TD on all, FD on top quality)
    td_features = [compute_td_features(seg.data) for seg in segments]
    td_indicators = compute_file_td_indicators(td_features)
    
    top_segments = select_top_quality_segments(segments, quality_report)
    fd_results = compute_envelope_analysis(top_segments, signal_data.fs, 
                                          config.rpm, config.envelope)
    
    # 4. Fuse and classify
    diagnosis = diagnose_bearing_fault(td_indicators, fd_results, 
                                     quality_report, config.fusion)
    health_result = classify_bearing_health(diagnosis, config.classification)
    
    return DiagnosisResult(
        health=health_result.health,
        fault_label=diagnosis.fault_label,
        confidence=diagnosis.confidence,
        rationale=diagnosis.rationale,
        quality=quality_report
    )
```

## Performance Targets and Validation

### Performance Metrics
- **Speed**: <2 seconds per file on standard hardware
- **Memory**: <100MB peak usage for typical files
- **Accuracy**: >95% on CWRU test set (with proper train/test split)

### Validation Strategy
1. **Unit Tests**: All feature computations validated against known results
2. **Golden Tests**: Frozen results on reference CWRU files
3. **Cross-validation**: Performance on held-out CWRU data
4. **Synthetic Tests**: Edge cases with known ground truth

## Migration Path from Current Implementation

1. **Phase 1**: Enhance existing modules with missing components
2. **Phase 2**: Add configuration system and CLI
3. **Phase 3**: Implement comprehensive testing
4. **Phase 4**: Performance optimization and caching
5. **Phase 5**: ML model integration framework

This architecture provides a solid foundation for the current rule-based approach while enabling future enhancements with machine learning models and additional sensors.