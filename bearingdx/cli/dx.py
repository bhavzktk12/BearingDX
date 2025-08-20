#!/usr/bin/env python3
"""
BearingDX Command Line Interface

Main CLI application for bearing fault diagnosis using vibration analysis.
Supports single file analysis, batch processing, and various output formats.

Usage Examples:
    # Single file classification
    python -m bearingdx.cli.dx classify data/bearing_001.mat --sensor DE --output results.json
    
    # Batch processing 
    python -m bearingdx.cli.dx batch data/ --pattern "*.mat" --output results.csv
    
    # Parse and inspect signal
    python -m bearingdx.cli.dx parse data/bearing_001.mat --show-segments --debug
    
    # Quality assessment
    python -m bearingdx.cli.dx quality data/bearing_001.mat --baseline normal_baseline.json
    
    # Configuration testing
    python -m bearingdx.cli.dx config --profile conservative --validate
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import pandas as pd
import numpy as np

# Import BearingDX modules (these would be implemented based on the interfaces)
try:
    from bearingdx.bdxio.loader import load_mat, detect_available_sensors
    from bearingdx.bdxio.segmenter import segment_signal, assess_segment_quality  
    from bearingdx.bdxio.cwru import get_fault_frequencies, infer_load_from_filename, get_rpm_from_load
    from bearingdx.features.td import compute_td_features, compute_file_td_indicators
    from bearingdx.features.fd import compute_envelope_analysis, compute_fft_ratios
    from bearingdx.features.quality import assess_signal_quality, build_quality_baselines
    from bearingdx.engine.fusion import diagnose_bearing_fault
    from bearingdx.engine.classify import classify_bearing_health
    from bearingdx.config.loader import load_config, validate_config
    from bearingdx.utils.logging import setup_logging
    from bearingdx.utils.validation import validate_input_file
    from bearingdx.utils.metrics import format_results
except ImportError as e:
    print(f"Error importing BearingDX modules: {e}")
    print("This is a demo CLI interface. Full implementation requires the complete module structure.")
    sys.exit(1)

# Version information
__version__ = "1.0.0"
__algorithm_version__ = "BDX-2024.1"

def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser with all subcommands."""
    
    parser = argparse.ArgumentParser(
        description="BearingDX - Vibration-based Bearing Fault Diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s classify bearing_data.mat --sensor DE --load 0
  %(prog)s batch data_folder/ --output results.csv --config conservative
  %(prog)s parse bearing_data.mat --debug --show-quality
  %(prog)s quality data/ --build-baseline --output baseline.json
        """
    )
    
    # Global options
    parser.add_argument('--version', action='version', version=f'BearingDX {__version__}')
    parser.add_argument('--config', type=str, default='defaults', 
                       help='Configuration profile (defaults, conservative, sensitive)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify bearing health from single file')
    classify_parser.add_argument('input', help='Input .mat file path')
    classify_parser.add_argument('--sensor', choices=['DE', 'FE', 'BA'], default='DE',
                                help='Primary sensor to analyze')
    classify_parser.add_argument('--load', type=int, choices=[0, 1, 2, 3],
                                help='Load condition (0-3 HP), auto-detected if not specified')
    classify_parser.add_argument('--fs', type=float, 
                                help='Sampling frequency override (Hz)')
    classify_parser.add_argument('--output', type=str,
                                help='Output file path (.json or .csv)')
    classify_parser.add_argument('--debug', action='store_true',
                                help='Include debug information in output')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Process multiple files')
    batch_parser.add_argument('input_dir', help='Input directory path')
    batch_parser.add_argument('--pattern', default='*.mat', 
                             help='File pattern to match (default: *.mat)')
    batch_parser.add_argument('--sensor', choices=['DE', 'FE', 'BA'], default='DE',
                             help='Primary sensor to analyze')
    batch_parser.add_argument('--output', required=True,
                             help='Output CSV file path')
    batch_parser.add_argument('--recursive', action='store_true',
                             help='Search subdirectories recursively')
    batch_parser.add_argument('--continue-on-error', action='store_true',
                             help='Continue processing if individual files fail')
    batch_parser.add_argument('--max-workers', type=int, default=1,
                             help='Number of parallel workers (1 = sequential)')
    
    # Parse command (inspection)
    parse_parser = subparsers.add_parser('parse', help='Parse and inspect file contents')
    parse_parser.add_argument('input', help='Input .mat file path')
    parse_parser.add_argument('--sensor', choices=['DE', 'FE', 'BA'], default='DE',
                             help='Primary sensor to analyze')
    parse_parser.add_argument('--show-segments', action='store_true',
                             help='Show segment-level information')
    parse_parser.add_argument('--show-quality', action='store_true',
                             help='Show quality assessment details')
    parse_parser.add_argument('--show-features', action='store_true', 
                             help='Show computed features')
    parse_parser.add_argument('--debug', action='store_true',
                             help='Show debug information')
    
    # Quality assessment command
    quality_parser = subparsers.add_parser('quality', help='Assess signal quality')
    quality_parser.add_argument('input', help='Input file or directory path')
    quality_parser.add_argument('--build-baseline', action='store_true',
                               help='Build quality baseline from Normal files')
    quality_parser.add_argument('--baseline', type=str,
                               help='Existing baseline file to use')
    quality_parser.add_argument('--sensor', choices=['DE', 'FE', 'BA'], default='DE',
                               help='Primary sensor to analyze')
    quality_parser.add_argument('--output', type=str,
                               help='Output baseline or report file')
    
    # Configuration command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--profile', type=str, default='defaults',
                              help='Configuration profile to show/validate')
    config_parser.add_argument('--validate', action='store_true',
                              help='Validate configuration')
    config_parser.add_argument('--show', action='store_true',
                              help='Show configuration values')
    config_parser.add_argument('--list-profiles', action='store_true',
                              help='List available configuration profiles')
    
    return parser

def process_single_file(file_path: str, config: Dict[str, Any], 
                       sensor: str = 'DE', load: Optional[int] = None,
                       fs_override: Optional[float] = None,
                       debug: bool = False) -> Dict[str, Any]:
    """
    Process a single .mat file and return diagnosis results.
    
    Args:
        file_path: Path to .mat file
        config: Configuration dictionary
        sensor: Primary sensor to analyze
        load: Load condition override
        fs_override: Sampling frequency override
        debug: Include debug information
        
    Returns:
        Dictionary with diagnosis results and metadata
    """
    start_time = time.time()
    
    try:
        # 1. Load and validate file
        logging.info(f"Processing file: {file_path}")
        signal_data = load_mat(file_path, sensor=sensor, fs_override=fs_override)
        
        # 2. Determine operating conditions
        if load is None:
            load = infer_load_from_filename(Path(file_path).name)
        if load is not None:
            rpm = get_rpm_from_load(load, config.get('load_rpm_map', {}))
        else:
            rpm = config['load_rpm_map'][0]  # Default to 0 HP
            
        # 3. Segment signal and assess quality
        segments = segment_signal(
            signal_data.signal, 
            signal_data.fs,
            window_size=config['segmentation']['window_size'],
            overlap=config['segmentation']['overlap'],
            normalization=config['segmentation']['normalization']
        )
        
        quality_report = assess_signal_quality(segments, config.get('quality_baselines'))
        
        # 4. Extract TD features
        td_features = [compute_td_features(seg.data) for seg in segments if seg.is_valid]
        td_indicators = compute_file_td_indicators(td_features, 
                                                  [seg.quality_score for seg in segments])
        
        # 5. Extract FD features (on top quality segments if enabled)
        if config['performance']['use_top_segments_only']:
            # Select top quality segments for FD analysis
            valid_segments = [seg for seg in segments if seg.is_valid]
            n_top = max(1, int(len(valid_segments) * config['performance']['top_segment_fraction']))
            top_segments = sorted(valid_segments, key=lambda s: s.quality_score, reverse=True)[:n_top]
            combined_signal = np.concatenate([seg.data for seg in top_segments])
        else:
            combined_signal = np.concatenate([seg.data for seg in segments if seg.is_valid])
            
        fd_results = compute_envelope_analysis(
            combined_signal,
            signal_data.fs,
            rpm,
            config['bearing_geometry'],
            config['envelope']['candidate_bands']
        )
        
        # 6. Diagnose fault
        diagnosis = diagnose_bearing_fault(
            td_indicators,
            fd_results, 
            quality_report,
            config['fusion_thresholds']
        )
        
        # 7. Classify health
        health = classify_bearing_health(diagnosis, config['classification'])
        
        # 8. Prepare results
        processing_time = time.time() - start_time
        
        results = {
            'file_path': file_path,
            'health': health,
            'fault_label': diagnosis.fault_label,
            'confidence': diagnosis.confidence,
            'processing_time': processing_time,
            'algorithm_version': __algorithm_version__,
            'sensor': sensor,
            'load': load,
            'rpm': rpm,
            'sampling_frequency': signal_data.fs,
            'quality': {
                'file_quality': quality_report.file_quality,
                'valid_segments': quality_report.valid_segments,
                'total_segments': quality_report.total_segments,
                'warnings': quality_report.overall_warnings
            },
            'rationale': diagnosis.rationale
        }
        
        if debug:
            results['debug'] = {
                'td_indicators': td_indicators.__dict__,
                'fd_results': fd_results.__dict__,
                'segments_info': {
                    'count': len(segments),
                    'valid_count': len([s for s in segments if s.is_valid]),
                    'mean_quality': np.mean([s.quality_score for s in segments])
                }
            }
            
        return results
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return {
            'file_path': file_path,
            'error': str(e),
            'health': 'Error',
            'fault_label': 'Error',
            'confidence': 0.0,
            'processing_time': time.time() - start_time
        }

def process_batch(input_dir: str, pattern: str, config: Dict[str, Any],
                 sensor: str = 'DE', recursive: bool = False,
                 continue_on_error: bool = True) -> List[Dict[str, Any]]:
    """
    Process multiple files in batch mode.
    
    Args:
        input_dir: Input directory path
        pattern: File pattern to match
        config: Configuration dictionary
        sensor: Primary sensor to analyze
        recursive: Search subdirectories
        continue_on_error: Continue if individual files fail
        
    Returns:
        List of results dictionaries
    """
    input_path = Path(input_dir)
    
    if recursive:
        files = list(input_path.rglob(pattern))
    else:
        files = list(input_path.glob(pattern))
        
    logging.info(f"Found {len(files)} files matching pattern '{pattern}'")
    
    results = []
    for i, file_path in enumerate(files):
        logging.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")
        
        try:
            result = process_single_file(str(file_path), config, sensor=sensor)
            results.append(result)
        except Exception as e:
            if continue_on_error:
                logging.error(f"Error processing {file_path}: {e}")
                results.append({
                    'file_path': str(file_path),
                    'error': str(e),
                    'health': 'Error',
                    'fault_label': 'Error',
                    'confidence': 0.0
                })
            else:
                raise
                
    return results

def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save results to JSON or CSV format based on file extension."""
    output_file = Path(output_path)
    
    if output_file.suffix.lower() == '.json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Results saved to JSON: {output_path}")
        
    elif output_file.suffix.lower() == '.csv':
        # Flatten results for CSV
        df_data = []
        for result in results:
            row = {
                'file_path': result.get('file_path', ''),
                'health': result.get('health', ''),
                'fault_label': result.get('fault_label', ''),
                'confidence': result.get('confidence', 0.0),
                'processing_time': result.get('processing_time', 0.0),
                'sensor': result.get('sensor', ''),
                'load': result.get('load', ''),
                'rpm': result.get('rpm', ''),
                'sampling_frequency': result.get('sampling_frequency', ''),
                'file_quality': result.get('quality', {}).get('file_quality', 0.0),
                'valid_segments': result.get('quality', {}).get('valid_segments', 0),
                'total_segments': result.get('quality', {}).get('total_segments', 0),
                'error': result.get('error', '')
            }
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        logging.info(f"Results saved to CSV: {output_path}")
    else:
        raise ValueError(f"Unsupported output format: {output_file.suffix}")

def main():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    if not args.command:
        parser.print_help()
        return
        
    try:
        # Load configuration
        config = load_config(profile=args.config)
        
        # Validate configuration
        validation_errors = validate_config(config)
        if validation_errors:
            logging.error("Configuration validation errors:")
            for error in validation_errors:
                logging.error(f"  - {error}")
            sys.exit(1)
            
        # Execute command
        if args.command == 'classify':
            result = process_single_file(
                args.input, 
                config,
                sensor=args.sensor,
                load=args.load,
                fs_override=args.fs,
                debug=args.debug
            )
            
            if args.output:
                save_results([result], args.output)
            else:
                print(json.dumps(result, indent=2, default=str))
                
        elif args.command == 'batch':
            results = process_batch(
                args.input_dir,
                args.pattern,
                config,
                sensor=args.sensor,
                recursive=args.recursive,
                continue_on_error=args.continue_on_error
            )
            
            save_results(results, args.output)
            
            # Print summary
            total = len(results)
            healthy = len([r for r in results if r.get('health') == 'Healthy'])
            unhealthy = len([r for r in results if r.get('health') == 'Unhealthy'])
            errors = len([r for r in results if r.get('error')])
            
            print(f"\nBatch processing complete:")
            print(f"  Total files: {total}")
            print(f"  Healthy: {healthy}")
            print(f"  Unhealthy: {unhealthy}")
            print(f"  Errors: {errors}")
            
        elif args.command == 'parse':
            # Implementation for parse command
            logging.info(f"Parsing file: {args.input}")
            # This would call the parsing functions with detailed output
            
        elif args.command == 'quality':
            # Implementation for quality command  
            logging.info(f"Quality assessment: {args.input}")
            # This would call quality assessment functions
            
        elif args.command == 'config':
            if args.list_profiles:
                print("Available configuration profiles:")
                print("  - defaults")
                print("  - conservative")
                print("  - sensitive")
            elif args.show:
                print(f"Configuration profile: {args.profile}")
                print(json.dumps(config, indent=2, default=str))
            elif args.validate:
                errors = validate_config(config)
                if errors:
                    print("Configuration validation errors:")
                    for error in errors:
                        print(f"  - {error}")
                else:
                    print("Configuration is valid.")
                    
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()