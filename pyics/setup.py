#!/usr/bin/env python3
"""
Pyics - Data-Oriented Calendar Automation System
Setup configuration with version isolation and modular architecture support

Author: OBINexus Engineering Team
License: MIT
Python: 3.8+
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

# Pyics version management
PYICS_VERSION = "2.1.0"
PYICS_MIN_PYTHON = "3.8"

# Project metadata
PROJECT_ROOT = Path(__file__).parent
README_PATH = PROJECT_ROOT / "README.md"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

def read_file(filepath):
    """Read file content with fallback handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def get_requirements():
    """Parse requirements.txt with version isolation support"""
    requirements = []
    try:
        with open(REQUIREMENTS_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        # Fallback requirements for core functionality
        requirements = [
            "click>=8.0.0",
            "pydantic>=1.8.0",
            "aiohttp>=3.8.0",
            "cryptography>=3.4.0",
            "python-dateutil>=2.8.0",
            "pytz>=2021.1",
            "email-validator>=1.1.0",
            "jinja2>=3.0.0",
            "pyyaml>=5.4.0",
            "python-jose[cryptography]>=3.3.0"
        ]
    return requirements

def get_optional_requirements():
    """Define optional dependency groups for modular installation"""
    return {
        # Core v1 dependencies
        'v1': [
            "click>=8.0.0",
            "python-dateutil>=2.8.0",
            "email-validator>=1.1.0"
        ],
        
        # Enhanced v2 dependencies
        'v2': [
            "aiohttp>=3.8.0",
            "cryptography>=3.4.0",
            "pydantic>=1.8.0",
            "python-jose[cryptography]>=3.3.0"
        ],
        
        # Experimental v3 dependencies
        'v3-preview': [
            "websockets>=10.0",
            "blockchain-py>=0.2.0",
            "openai>=0.27.0"
        ],
        
        # Development and testing
        'dev': [
            "pytest>=6.2.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0"
        ],
        
        # Documentation
        'docs': [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0"
        ],
        
        # Enterprise features
        'enterprise': [
            "ldap3>=2.9.0",
            "redis>=3.5.0",
            "prometheus-client>=0.11.0",
            "opentelemetry-api>=1.1.0"
        ],
        
        # Distribution and networking
        'distribute': [
            "celery>=5.2.0",
            "kombu>=5.2.0",
            "requests>=2.25.0"
        ]
    }

class PostDevelopCommand(develop):
    """Post-installation setup for development"""
    def run(self):
        develop.run(self)
        print("Setting up Pyics development environment...")
        self._setup_cli_registration()
        self._validate_version_isolation()

class PostInstallCommand(install):
    """Post-installation setup for production"""
    def run(self):
        install.run(self)
        print("Configuring Pyics production environment...")
        self._setup_cli_registration()

    def _setup_cli_registration(self):
        """Register CLI commands with IoC container"""
        try:
            from pyics.v1.cli.ioc import register_commands
            register_commands()
            print("✅ CLI commands registered successfully")
        except ImportError:
            print("⚠️  CLI registration skipped - run 'pyics setup' after installation")

    def _validate_version_isolation(self):
        """Validate version isolation integrity"""
        version_paths = ['v1', 'v2', 'v3-preview']
        for version in version_paths:
            version_path = PROJECT_ROOT / 'pyics' / version
            if version_path.exists():
                print(f"✅ Version {version} structure validated")
            else:
                print(f"⚠️  Version {version} not found - partial installation")

def get_console_scripts():
    """Define CLI entry points with version awareness"""
    return [
        # Main CLI interface
        'pyics=pyics.cli.main:main',
        
        # Version-specific interfaces
        'pyics-v1=pyics.v1.cli.main:main',
        'pyics-v2=pyics.v2.cli.main:main',
        'pyics-v3=pyics.v3_preview.cli.main:main',
        
        # Specialized commands
        'pyics-audit=pyics.audit.cli:main',
        'pyics-notify=pyics.notify.cli:main',
        'pyics-calendar=pyics.calendar.cli:main',
        
        # REPL interface
        'pyics-repl=pyics.cli.repl:main',
        
        # Migration tools
        'pyics-migrate=pyics.migration.cli:main'
    ]

def validate_python_version():
    """Ensure compatible Python version"""
    if sys.version_info < tuple(map(int, PYICS_MIN_PYTHON.split('.'))):
        raise SystemExit(
            f"Pyics requires Python {PYICS_MIN_PYTHON}+ "
            f"(current: {sys.version_info.major}.{sys.version_info.minor})"
        )

# Validate environment before setup
validate_python_version()

# Package discovery with namespace support
packages = find_packages(
    where='.',
    include=['pyics*'],
    exclude=['tests*', 'docs*', 'scripts*']
)

# Dynamic classifiers based on development status
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: System Administrators",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Office/Business :: Scheduling",
    "Topic :: Communications :: Email",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Distributed Computing",
    "Framework :: AsyncIO"
]

# Setup configuration
setup(
    name="pyics",
    version=PYICS_VERSION,
    author="OBINexus Engineering Team",
    author_email="engineering@obinexus.org",
    description="Data-Oriented Calendar Automation System with Version Isolation",
    long_description=read_file(README_PATH),
    long_description_content_type="text/markdown",
    url="https://github.com/obinexus/pyics",
    project_urls={
        "Bug Reports": "https://github.com/obinexus/pyics/issues",
        "Source": "https://github.com/obinexus/pyics",
        "Documentation": "https://pyics.readthedocs.io",
        "Funding": "https://github.com/sponsors/obinexus"
    },
    
    # Package configuration
    packages=packages,
    package_dir={'': '.'},
    include_package_data=True,
    package_data={
        'pyics': [
            'config/*.json',
            'config/schemas/*.json',
            'config/templates/*.json',
            'docs/api/schemas/*.json',
            'docs/api/openapi/*.yaml'
        ]
    },
    
    # Dependencies
    python_requires=f">={PYICS_MIN_PYTHON}",
    install_requires=get_requirements(),
    extras_require=get_optional_requirements(),
    
    # CLI configuration
    entry_points={
        'console_scripts': get_console_scripts(),
        'pyics.plugins': [
            'v1_core=pyics.v1.core:CorePlugin',
            'v2_enhanced=pyics.v2.core:EnhancedPlugin',
            'audit=pyics.audit:AuditPlugin',
            'calendar=pyics.calendar:CalendarPlugin',
            'notify=pyics.notify:NotificationPlugin'
        ]
    },
    
    # Metadata
    classifiers=classifiers,
    keywords="calendar ics automation data-oriented-programming enterprise cli",
    license="MIT",
    zip_safe=False,
    
    # Custom commands
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand
    },
    
    # Platform-specific configurations
    platforms=['any'],
    
    # Testing configuration
    test_suite='tests',
    tests_require=[
        'pytest>=6.2.0',
        'pytest-asyncio>=0.18.0',
        'pytest-cov>=2.12.0'
    ]
)

# Post-setup validation
if __name__ == "__main__":
    print(f"""
🚀 Pyics {PYICS_VERSION} Setup Complete!

Next steps:
1. Install development dependencies: pip install -e ".[dev]"
2. Initialize CLI registry: pyics setup
3. Validate installation: pyics version
4. Start REPL: pyics-repl
5. View documentation: pyics docs

Architecture Features:
✅ Version isolation (v1/v2/v3-preview)
✅ Data-oriented programming support
✅ Lambda calculus function composition
✅ IoC container with CLI registration
✅ Modular plugin architecture
✅ Enterprise-grade authentication
✅ Distributed calendar automation

For enterprise features: pip install -e ".[enterprise,distribute]"
For development setup: pip install -e ".[dev,docs]"
""")
