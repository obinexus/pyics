#!/usr/bin/env python3
"""
setup.py
Pyics Package Configuration - Corrected Structure

Generated: 2025-05-31T19:41:43.766539
Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Proper Python package configuration with root-level setup.py placement

PROBLEM SOLVED: Establishes standard Python package structure with proper discovery
DEPENDENCIES: setuptools, find_packages for automatic package discovery
PACKAGE STRUCTURE: pyics/ as installable package with config module inclusion
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

# Package metadata
setup(
    name="pyics",
    version="1.0.0",
    description="Data-Oriented Calendar Automation System with Version Isolation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OBINexus Engineering Team",
    author_email="engineering@obinexus.org",
    url="https://github.com/obinexus/pyics",
    
    # Package discovery configuration
    packages=find_packages(include=["pyics", "pyics.*"]),
    package_dir={"": "."},
    
    # Include configuration modules
    package_data={
        "pyics.config": ["*.json", "*.yaml", "*.toml"],
        "pyics.core": ["*.py"],
        "pyics.cli": ["*.py"],
    },
    include_package_data=True,
    
    # Python version requirements
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "click>=8.0.0",
        "pydantic>=1.8.0",
        "typing-extensions>=4.0.0",
        "python-dateutil>=2.8.0",
        "icalendar>=4.0.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "mypy>=0.910",
            "flake8>=3.8.0",
        ],
        "enterprise": [
            "cryptography>=3.4.0",
            "ldap3>=2.9.0",
            "oauth2lib>=0.1.0",
        ],
        "telemetry": [
            "opentelemetry-api>=1.0.0",
            "opentelemetry-sdk>=1.0.0",
            "prometheus-client>=0.11.0",
        ],
    },
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "pyics=pyics.cli.main:main",
            "pyics-validate=pyics.cli.validation.main:validation_cli",
            "pyics-generate=pyics.cli.composition.main:composition_cli",
            "pyics-audit=pyics.cli.registry.main:registry_cli",
        ],
    },
    
    # Classification metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Scheduling",
        "Topic :: System :: Systems Administration",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    
    # Keywords for package discovery
    keywords=[
        "calendar", "automation", "scheduling", "icalendar", "ics",
        "data-oriented-programming", "domain-driven-design", "enterprise",
        "compliance", "audit", "business-logic", "functional-programming"
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://pyics.readthedocs.io/",
        "Source": "https://github.com/obinexuscomputing/pyics",
        "Tracker": "https://github.com/obinexuscomputing/pyics/issues",
        "Changelog": "https://github.com/obinexuscomputing/pyics/blob/main/CHANGELOG.md",
    },
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Ensure pyics.config is discoverable
    namespace_packages=[],
)

# [EOF] - End of corrected setup.py configuration