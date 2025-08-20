"""
BearingDX - Vibration-Based Bearing Fault Diagnosis

A comprehensive tool for bearing fault diagnosis using vibration analysis
with focus on the CWRU (Case Western Reserve University) bearing dataset.

Key Features:
- Time-domain and frequency-domain feature extraction
- Envelope analysis with automatic band selection  
- Multi-sensor fusion (DE, FE, BA)
- Quality-aware processing and filtering
- Configurable conservative/sensitive detection profiles
- Production-ready CLI and batch processing

Basic Usage:
    import bearingdx as bdx
    
    # Load configuration
    config = bdx.load_config('conservative')
    
    # Diagnose single file
    result = bdx.diagnose_bearing('file.mat', config)
    print(f"Health: {result.health}, Fault: {result.fault_label}")

For detailed examples, see the documentation and example notebooks.
"""

# Version information
__version__ = "1.0.0"
__algorithm_version__ = "BDX-2024.1"
__author__ = "BearingDX Development Team"
__email__ = "bearingdx@example.com"

# Core imports for public API
try:
    from .config.loader import load_config, validate_config
    from .bdxio.loader import load_mat
    from .engine.fusion import diagnose_bearing_fault as diagnose_bearing
    from .engine.classify import classify_bearing_health
    
    # Main processing functions
    from .api import (
        process_single_file,
        process_batch,
        build_quality_baseline,
    )
    
except ImportError:
    # Handle case where modules are not yet implemented
    import warnings
    warnings.warn(
        "BearingDX modules not fully implemented. "
        "This is a architecture proposal with interface definitions.",
        ImportWarning
    )
    
    # Provide stub functions for demonstration
    def load_config(profile='defaults'):
        """Load configuration profile (stub implementation)."""
        return {'profile': profile, 'version': __version__}
    
    def diagnose_bearing(file_path, config=None):
        """Diagnose bearing fault (stub implementation)."""
        from collections import namedtuple
        Result = namedtuple('Result', ['health', 'fault_label', 'confidence'])
        return Result('Healthy', 'Normal', 0.85)

# Public API exports
__all__ = [
    # Version information
    '__version__',
    '__algorithm_version__',
    '__author__',
    '__email__',
    
    # Configuration
    'load_config',
    'validate_config',
    
    # Core processing
    'load_mat',
    'diagnose_bearing',
    'classify_bearing_health',
    
    # High-level API
    'process_single_file', 
    'process_batch',
    'build_quality_baseline',
    
    # Data structures (imported from submodules)
    'BearingGeometry',
    'DiagnosisResult',
    'ProcessingConfig',
]

# Module-level configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Compatibility information
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("BearingDX requires Python 3.8 or later")

def get_version_info():
    """Get detailed version information."""
    import platform
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "not installed"
        
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = "not installed"
        
    return {
        'bearingdx': __version__,
        'algorithm': __algorithm_version__,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'numpy': numpy_version,
        'scipy': scipy_version,
    }

def print_version_info():
    """Print detailed version information."""
    info = get_version_info()
    print(f"BearingDX {info['bearingdx']} (Algorithm: {info['algorithm']})")
    print(f"Python {info['python']} on {info['platform']}")
    print(f"Dependencies: NumPy {info['numpy']}, SciPy {info['scipy']}")

# Configuration for development/testing
_DEVELOPMENT_MODE = False

def set_development_mode(enabled=True):
    """Enable/disable development mode with additional debugging."""
    global _DEVELOPMENT_MODE
    _DEVELOPMENT_MODE = enabled
    
    if enabled:
        logging.basicConfig(level=logging.DEBUG)
        print(f"BearingDX development mode enabled")
    else:
        logging.basicConfig(level=logging.INFO)

def is_development_mode():
    """Check if development mode is enabled."""
    return _DEVELOPMENT_MODE