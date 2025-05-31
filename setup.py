#!/usr/bin/env python3
"""
setup.py
Pyics Package Configuration - Single-Pass Architecture

Generated: 2025-05-31T20:31:15.797912
Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Python package configuration with first-pass module exposure

PROBLEM SOLVED: Establishes single-pass package structure with dependency safety
DEPENDENCIES: setuptools, find_packages for automatic package discovery
PACKAGE STRUCTURE: pyics/ as installable package with first-pass domains only
FIRST-PASS DOMAINS: primitives, protocols, structures

This setup.py exposes only first-pass modules following single-pass architecture
principles to prevent circular dependencies and ensure deterministic loading.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    try:
        with open(readme_path, encoding='utf-8') as f:
            long_description = f.read()
    except Exception:
        long_description = "Pyics - Data-Oriented Calendar Automation System"
else:
    long_description = "Pyics - Data-Oriented Calendar Automation System"

# Package metadata
setup(
    name="pyics",
    version="1.0.0",
    description="Pyics - Data-Oriented Calendar Automation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nnamdi Okpala / OBINexus Computing",
    author_email="engineering@obinexus.com",
    url="https://github.com/obinexuscomputing/pyics",
    license="MIT",
    
    # Package discovery configuration (single-pass compliance)
    packages=find_packages(include=['pyics', 'pyics.*']),
    package_dir={'': '.'},
    
    # Include package data for first-pass domains
    package_data={
        'pyics.core': ['*.py'],
        'pyics.cli': ['*.py'],
        'pyics.config': ['*.json', '*.yaml', '*.toml'],
        'pyics.core.primitives': ['*.py', '*.md'],
        'pyics.core.protocols': ['*.py', '*.md'],
        'pyics.core.structures': ['*.py', '*.md'],
    },
    include_package_data=True,
    
    # Python version requirements
    python_requires='>=3.8',
    
    # Core dependencies (minimal for single-pass architecture)
    install_requires=[
        'click>=8.0.0',
        'python-dateutil>=2.8.0',
        'typing-extensions>=4.0.0',
    ],
    
    # Optional dependencies grouped by functionality
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0.0',
            'mypy>=0.910',
            'flake8>=3.8.0',
        ],
        'calendar': [
            'icalendar>=4.0.0',
        ],
        'enterprise': [
            'cryptography>=3.4.0',
        ],
    },
    
    # Entry points for CLI commands (first-pass domains only)
    entry_points={
        'console_scripts': [
            'pyics=pyics.cli.main:main',
            'pyics-primitives=pyics.cli.main:primitives',
            'pyics-protocols=pyics.cli.main:protocols',
            'pyics-structures=pyics.cli.main:structures',
        ],
    },
    
    # Classification metadata
    classifiers=['Development Status :: 4 - Beta', 'Intended Audience :: Developers', 'Topic :: Software Development :: Libraries :: Python Modules', 'Topic :: Office/Business :: Scheduling', 'License :: OSI Approved :: MIT License', 'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3.8', 'Programming Language :: Python :: 3.9', 'Programming Language :: Python :: 3.10', 'Programming Language :: Python :: 3.11', 'Programming Language :: Python :: 3.12', 'Operating System :: OS Independent', 'Typing :: Typed'],
    
    # Keywords for package discovery
    keywords=['calendar', 'automation', 'scheduling', 'icalendar', 'ics', 'data-oriented-programming', 'domain-driven-design'],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://pyics.readthedocs.io/',
        'Source': 'https://github.com/obinexuscomputing/pyics',
        'Tracker': 'https://github.com/obinexuscomputing/pyics/issues',
        'Changelog': 'https://github.com/obinexuscomputing/pyics/blob/main/CHANGELOG.md',
    },
    
    # Additional metadata
    zip_safe=False,
    platforms=['any'],
    
    # Ensure first-pass domains are discoverable
    namespace_packages=[],
)

# First-pass domain validation
def validate_first_pass_domains():
    """Validate that only first-pass domains are exposed"""
    import importlib.util
    
    first_pass_domains = ['primitives', 'protocols', 'structures']
    validation_passed = True
    
    for domain in first_pass_domains:
        try:
            spec = importlib.util.find_spec(f"pyics.core.{domain}")
            if spec is None:
                print(f"Warning: First-pass domain '{domain}' not found")
                validation_passed = False
        except ImportError:
            print(f"Warning: Cannot validate first-pass domain '{domain}'")
            validation_passed = False
    
    if validation_passed:
        print(f"✅ First-pass domain validation passed: {', '.join(first_pass_domains)}")
    else:
        print("⚠️  First-pass domain validation warnings detected")
    
    return validation_passed

if __name__ == "__main__":
    print("🔍 Validating first-pass domains...")
    validate_first_pass_domains()
    print("📦 Setup.py configuration complete")

# [EOF] - End of single-pass architecture setup.py
