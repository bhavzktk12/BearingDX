#!/usr/bin/env python3
"""
BearingDX Setup Script

Installation script for the BearingDX vibration-based bearing diagnostics package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "BearingDX - Vibration-based Bearing Fault Diagnosis Tool"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
        "click>=8.0.0",
        "tqdm>=4.60.0"
    ]

setup(
    name="bearingdx",
    version="1.0.0",
    description="Vibration-based bearing fault diagnosis using CWRU dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="BearingDX Development Team",
    author_email="bearingdx@example.com",
    url="https://github.com/bearingdx/bearingdx",
    
    # Package configuration
    packages=find_packages(include=["bearingdx", "bearingdx.*"]),
    include_package_data=True,
    package_data={
        "bearingdx": [
            "config/*.yaml",
            "config/*.json",
        ]
    },
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "performance": [
            "numba>=0.56.0",
            "joblib>=1.1.0",
        ],
        "advanced": [
            "pywavelets>=1.1.1",
            "librosa>=0.8.1",
        ]
    },
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "bearingdx=bearingdx.cli.dx:main",
            "bdx=bearingdx.cli.dx:main",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classification metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for searchability
    keywords=[
        "bearing",
        "fault diagnosis", 
        "vibration analysis",
        "condition monitoring",
        "predictive maintenance",
        "signal processing",
        "CWRU dataset",
        "envelope analysis",
        "time domain",
        "frequency domain",
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/bearingdx/bearingdx/issues",
        "Source": "https://github.com/bearingdx/bearingdx",
        "Documentation": "https://bearingdx.readthedocs.io/",
    },
    
    # License
    license="MIT",
    
    # Zip safety
    zip_safe=False,
)