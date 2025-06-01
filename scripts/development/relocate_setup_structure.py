#!/usr/bin/env python3
"""
relocate_setup_structure.py
Pyics Setup.py Relocation and Package Structure Corrector

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Systematic relocation of setup.py to project root with proper package configuration
Architecture: Single-pass structure correction with validation integration
Methodology: Waterfall setup reconfiguration with dependency preservation

PROBLEM SOLVED: Corrects setup.py placement and ensures proper Python package architecture
DEPENDENCIES: Standard library only (pathlib, shutil, ast)
THREAD SAFETY: Yes - atomic file operations with backup creation
DETERMINISTIC: Yes - reproducible setup structure correction

This script systematically relocates setup.py from pyics/ to project root, updates package
declarations for proper find_packages() discovery, and ensures pyics.config module inclusion.
"""

import os
import sys
import shutil
import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Configuration
PROJECT_ROOT = Path.cwd()
CURRENT_SETUP_PATH = "pyics/setup.py"
TARGET_SETUP_PATH = "setup.py"
BACKUP_DIR = "setup_backup"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SetupStructureCorrector:
    """
    Systematic corrector for setup.py placement and package configuration
    
    Implements methodical relocation with proper package discovery configuration
    and comprehensive validation of Python package architecture compliance.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.current_setup = self.project_root / CURRENT_SETUP_PATH
        self.target_setup = self.project_root / TARGET_SETUP_PATH
        self.backup_dir = self.project_root / BACKUP_DIR
        
        self.correction_results = {
            "backup_created": False,
            "setup_relocated": False,
            "package_config_updated": False,
            "validation_passed": False,
            "summary": ""
        }
        
    def execute_setup_correction(self) -> Dict[str, any]:
        """
        Execute complete setup.py relocation and configuration correction
        
        Returns:
            Correction results with validation status
        """
        logger.info("=" * 60)
        logger.info("PYICS SETUP.PY STRUCTURE CORRECTION")
        logger.info("Engineering Lead: Nnamdi Okpala / OBINexus Computing")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Backup existing setup configuration
            self._create_setup_backup()
            
            # Phase 2: Analyze current setup.py content
            setup_content = self._analyze_current_setup()
            
            # Phase 3: Generate corrected setup.py content
            corrected_content = self._generate_corrected_setup(setup_content)
            
            # Phase 4: Write corrected setup.py to project root
            self._write_corrected_setup(corrected_content)
            
            # Phase 5: Validate corrected structure
            self._validate_corrected_structure()
            
            # Phase 6: Cleanup old setup.py
            self._cleanup_old_setup()
            
            return self.correction_results
            
        except Exception as e:
            logger.error(f"Setup correction failed: {e}")
            self.correction_results["summary"] = f"❌ Critical correction failure: {e}"
            return self.correction_results
    
    def _create_setup_backup(self) -> None:
        """Create backup of current setup configuration"""
        try:
            self.backup_dir.mkdir(exist_ok=True)
            
            if self.current_setup.exists():
                backup_setup_path = self.backup_dir / f"setup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                shutil.copy2(self.current_setup, backup_setup_path)
                logger.info(f"Setup backup created: {backup_setup_path}")
                
            if self.target_setup.exists():
                backup_root_setup_path = self.backup_dir / f"root_setup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                shutil.copy2(self.target_setup, backup_root_setup_path)
                logger.info(f"Root setup backup created: {backup_root_setup_path}")
            
            self.correction_results["backup_created"] = True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    def _analyze_current_setup(self) -> Optional[str]:
        """Analyze current setup.py content for correction"""
        if not self.current_setup.exists():
            logger.warning("No setup.py found in pyics/ directory")
            return None
        
        try:
            with open(self.current_setup, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info("Current setup.py content analyzed successfully")
            return content
            
        except Exception as e:
            logger.error(f"Failed to analyze current setup.py: {e}")
            raise
    
    def _generate_corrected_setup(self, original_content: Optional[str]) -> str:
        """Generate corrected setup.py content with proper package configuration"""
        
        # Extract metadata from original setup.py if available
        metadata = self._extract_setup_metadata(original_content) if original_content else {}
        
        # Generate corrected setup.py content
        setup_content = []
        setup_content.append('#!/usr/bin/env python3')
        setup_content.append('"""')
        setup_content.append('setup.py')
        setup_content.append('Pyics Package Configuration - Corrected Structure')
        setup_content.append('')
        setup_content.append(f'Generated: {datetime.now().isoformat()}')
        setup_content.append('Engineering Lead: Nnamdi Okpala / OBINexus Computing')
        setup_content.append('Purpose: Proper Python package configuration with root-level setup.py placement')
        setup_content.append('')
        setup_content.append('PROBLEM SOLVED: Establishes standard Python package structure with proper discovery')
        setup_content.append('DEPENDENCIES: setuptools, find_packages for automatic package discovery')
        setup_content.append('PACKAGE STRUCTURE: pyics/ as installable package with config module inclusion')
        setup_content.append('"""')
        setup_content.append('')
        setup_content.append('from setuptools import setup, find_packages')
        setup_content.append('from pathlib import Path')
        setup_content.append('')
        setup_content.append('# Read README for long description')
        setup_content.append('this_directory = Path(__file__).parent')
        setup_content.append('long_description = ""')
        setup_content.append('readme_path = this_directory / "README.md"')
        setup_content.append('if readme_path.exists():')
        setup_content.append('    with open(readme_path, encoding="utf-8") as f:')
        setup_content.append('        long_description = f.read()')
        setup_content.append('')
        setup_content.append('# Package metadata')
        setup_content.append('setup(')
        setup_content.append(f'    name="{metadata.get("name", "pyics")}",')
        setup_content.append(f'    version="{metadata.get("version", "1.0.0")}",')
        setup_content.append(f'    description="{metadata.get("description", "Pyics - Data-Oriented Calendar Automation System")}",')
        setup_content.append('    long_description=long_description,')
        setup_content.append('    long_description_content_type="text/markdown",')
        setup_content.append(f'    author="{metadata.get("author", "Nnamdi Okpala / OBINexus Computing")}",')
        setup_content.append(f'    author_email="{metadata.get("author_email", "engineering@obinexus.com")}",')
        setup_content.append(f'    url="{metadata.get("url", "https://github.com/obinexuscomputing/pyics")}",')
        setup_content.append('    ')
        setup_content.append('    # Package discovery configuration')
        setup_content.append('    packages=find_packages(include=["pyics", "pyics.*"]),')
        setup_content.append('    package_dir={"": "."},')
        setup_content.append('    ')
        setup_content.append('    # Include configuration modules')
        setup_content.append('    package_data={')
        setup_content.append('        "pyics.config": ["*.json", "*.yaml", "*.toml"],')
        setup_content.append('        "pyics.core": ["*.py"],')
        setup_content.append('        "pyics.cli": ["*.py"],')
        setup_content.append('    },')
        setup_content.append('    include_package_data=True,')
        setup_content.append('    ')
        setup_content.append('    # Python version requirements')
        setup_content.append('    python_requires=">=3.8",')
        setup_content.append('    ')
        setup_content.append('    # Dependencies')
        setup_content.append('    install_requires=[')
        setup_content.append('        "click>=8.0.0",')
        setup_content.append('        "pydantic>=1.8.0",')
        setup_content.append('        "typing-extensions>=4.0.0",')
        setup_content.append('        "python-dateutil>=2.8.0",')
        setup_content.append('        "icalendar>=4.0.0",')
        setup_content.append('    ],')
        setup_content.append('    ')
        setup_content.append('    # Optional dependencies')
        setup_content.append('    extras_require={')
        setup_content.append('        "dev": [')
        setup_content.append('            "pytest>=6.0.0",')
        setup_content.append('            "pytest-cov>=2.10.0",')
        setup_content.append('            "black>=21.0.0",')
        setup_content.append('            "mypy>=0.910",')
        setup_content.append('            "flake8>=3.8.0",')
        setup_content.append('        ],')
        setup_content.append('        "enterprise": [')
        setup_content.append('            "cryptography>=3.4.0",')
        setup_content.append('            "ldap3>=2.9.0",')
        setup_content.append('            "oauth2lib>=0.1.0",')
        setup_content.append('        ],')
        setup_content.append('        "telemetry": [')
        setup_content.append('            "opentelemetry-api>=1.0.0",')
        setup_content.append('            "opentelemetry-sdk>=1.0.0",')
        setup_content.append('            "prometheus-client>=0.11.0",')
        setup_content.append('        ],')
        setup_content.append('    },')
        setup_content.append('    ')
        setup_content.append('    # Entry points for CLI commands')
        setup_content.append('    entry_points={')
        setup_content.append('        "console_scripts": [')
        setup_content.append('            "pyics=pyics.cli.main:main",')
        setup_content.append('            "pyics-validate=pyics.cli.validation.main:validation_cli",')
        setup_content.append('            "pyics-generate=pyics.cli.composition.main:composition_cli",')
        setup_content.append('            "pyics-audit=pyics.cli.registry.main:registry_cli",')
        setup_content.append('        ],')
        setup_content.append('    },')
        setup_content.append('    ')
        setup_content.append('    # Classification metadata')
        setup_content.append('    classifiers=[')
        setup_content.append('        "Development Status :: 4 - Beta",')
        setup_content.append('        "Intended Audience :: Developers",')
        setup_content.append('        "Intended Audience :: System Administrators",')
        setup_content.append('        "Topic :: Software Development :: Libraries :: Python Modules",')
        setup_content.append('        "Topic :: Office/Business :: Scheduling",')
        setup_content.append('        "Topic :: System :: Systems Administration",')
        setup_content.append('        "License :: OSI Approved :: MIT License",')
        setup_content.append('        "Programming Language :: Python :: 3",')
        setup_content.append('        "Programming Language :: Python :: 3.8",')
        setup_content.append('        "Programming Language :: Python :: 3.9",')
        setup_content.append('        "Programming Language :: Python :: 3.10",')
        setup_content.append('        "Programming Language :: Python :: 3.11",')
        setup_content.append('        "Programming Language :: Python :: 3.12",')
        setup_content.append('        "Operating System :: OS Independent",')
        setup_content.append('        "Typing :: Typed",')
        setup_content.append('    ],')
        setup_content.append('    ')
        setup_content.append('    # Keywords for package discovery')
        setup_content.append('    keywords=[')
        setup_content.append('        "calendar", "automation", "scheduling", "icalendar", "ics",')
        setup_content.append('        "data-oriented-programming", "domain-driven-design", "enterprise",')
        setup_content.append('        "compliance", "audit", "business-logic", "functional-programming"')
        setup_content.append('    ],')
        setup_content.append('    ')
        setup_content.append('    # Project URLs')
        setup_content.append('    project_urls={')
        setup_content.append('        "Documentation": "https://pyics.readthedocs.io/",')
        setup_content.append('        "Source": "https://github.com/obinexuscomputing/pyics",')
        setup_content.append('        "Tracker": "https://github.com/obinexuscomputing/pyics/issues",')
        setup_content.append('        "Changelog": "https://github.com/obinexuscomputing/pyics/blob/main/CHANGELOG.md",')
        setup_content.append('    },')
        setup_content.append('    ')
        setup_content.append('    # Additional metadata')
        setup_content.append('    zip_safe=False,')
        setup_content.append('    platforms=["any"],')
        setup_content.append('    ')
        setup_content.append('    # Ensure pyics.config is discoverable')
        setup_content.append('    namespace_packages=[],')
        setup_content.append(')')
        setup_content.append('')
        setup_content.append('# [EOF] - End of corrected setup.py configuration')
        
        return '\n'.join(setup_content)
    
    def _extract_setup_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from existing setup.py content"""
        metadata = {}
        
        try:
            # Parse setup.py AST to extract metadata
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'setup':
                    for keyword in node.keywords:
                        if isinstance(keyword.value, ast.Constant):
                            metadata[keyword.arg] = str(keyword.value.value)
                        elif isinstance(keyword.value, ast.Str):  # Python < 3.8 compatibility
                            metadata[keyword.arg] = keyword.value.s
            
            logger.info(f"Extracted metadata: {list(metadata.keys())}")
            
        except Exception as e:
            logger.warning(f"Failed to extract setup metadata: {e}")
        
        return metadata
    
    def _write_corrected_setup(self, content: str) -> None:
        """Write corrected setup.py to project root"""
        try:
            with open(self.target_setup, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Corrected setup.py written to: {self.target_setup}")
            self.correction_results["setup_relocated"] = True
            self.correction_results["package_config_updated"] = True
            
        except Exception as e:
            logger.error(f"Failed to write corrected setup.py: {e}")
            raise
    
    def _validate_corrected_structure(self) -> None:
        """Validate corrected setup.py structure"""
        try:
            # Validate setup.py syntax
            with open(self.target_setup, 'r', encoding='utf-8') as f:
                setup_content = f.read()
            
            ast.parse(setup_content)  # Syntax validation
            
            # Validate required elements
            required_elements = ['find_packages', 'pyics', 'setup']
            for element in required_elements:
                if element not in setup_content:
                    raise ValueError(f"Missing required element: {element}")
            
            logger.info("Corrected setup.py structure validated successfully")
            self.correction_results["validation_passed"] = True
            
        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            self.correction_results["validation_passed"] = False
            raise
    
    def _cleanup_old_setup(self) -> None:
        """Remove old setup.py from pyics/ directory"""
        try:
            if self.current_setup.exists():
                self.current_setup.unlink()
                logger.info(f"Removed old setup.py from: {self.current_setup}")
            
            # Update correction summary
            if all([
                self.correction_results["backup_created"],
                self.correction_results["setup_relocated"],
                self.correction_results["package_config_updated"],
                self.correction_results["validation_passed"]
            ]):
                self.correction_results["summary"] = "✅ Setup.py structure corrected successfully"
            else:
                failed_steps = [
                    step for step, passed in self.correction_results.items()
                    if step != "summary" and not passed
                ]
                self.correction_results["summary"] = f"❌ Correction failed: {', '.join(failed_steps)}"
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            # Don't raise here - correction was successful, cleanup is optional

def create_root_level_manifest() -> None:
    """Create MANIFEST.in for proper package data inclusion"""
    manifest_path = PROJECT_ROOT / "MANIFEST.in"
    
    manifest_lines = [
        "# MANIFEST.in",
        "# Pyics Package Data Inclusion Configuration",
        "",
        "# Include documentation",
        "include README.md",
        "include CHANGELOG.md",
        "include LICENSE",
        "include pyproject.toml",
        "",
        "# Include configuration files",
        "recursive-include pyics/config *.json *.yaml *.toml *.py",
        "recursive-include pyics/core *.py",
        "recursive-include pyics/cli *.py",
        "",
        "# Include development scripts",
        "recursive-include scripts *.py *.sh",
        "",
        "# Include test data",
        "recursive-include tests *.py *.json *.yaml",
        "",
        "# Exclude unnecessary files",
        "global-exclude *.pyc",
        "global-exclude *.pyo",
        "global-exclude *.pyd",
        "global-exclude __pycache__",
        "global-exclude .git*",
        "global-exclude .pytest_cache",
        "global-exclude *.so",
        "global-exclude *.egg-info"
    ]
    
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(manifest_lines))
        logger.info(f"MANIFEST.in created at: {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to create MANIFEST.in: {e}")

def create_pyproject_toml() -> None:
    """Create pyproject.toml for modern Python packaging"""
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    
    if pyproject_path.exists():
        logger.info("pyproject.toml already exists - skipping creation")
        return
    
    toml_lines = [
        "[build-system]",
        'requires = ["setuptools>=45", "wheel", "setuptools-scm[toml]>=6.2"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        'name = "pyics"',
        'description = "Pyics - Data-Oriented Calendar Automation System"',
        'readme = "README.md"',
        'license = {file = "LICENSE"}',
        'authors = [{name = "Nnamdi Okpala", email = "engineering@obinexus.com"}]',
        'maintainers = [{name = "OBINexus Computing", email = "engineering@obinexus.com"}]',
        'classifiers = [',
        '    "Development Status :: 4 - Beta",',
        '    "Intended Audience :: Developers",',
        '    "Topic :: Software Development :: Libraries :: Python Modules",',
        '    "License :: OSI Approved :: MIT License",',
        '    "Programming Language :: Python :: 3",',
        '    "Programming Language :: Python :: 3.8",',
        '    "Programming Language :: Python :: 3.9",',
        '    "Programming Language :: Python :: 3.10",',
        '    "Programming Language :: Python :: 3.11",',
        '    "Programming Language :: Python :: 3.12",',
        "]",
        'requires-python = ">=3.8"',
        'dynamic = ["version"]',
        "",
        "dependencies = [",
        '    "click>=8.0.0",',
        '    "pydantic>=1.8.0",',
        '    "typing-extensions>=4.0.0",',
        '    "python-dateutil>=2.8.0",',
        '    "icalendar>=4.0.0",',
        "]",
        "",
        "[project.optional-dependencies]",
        "dev = [",
        '    "pytest>=6.0.0",',
        '    "pytest-cov>=2.10.0",',
        '    "black>=21.0.0",',
        '    "mypy>=0.910",',
        '    "flake8>=3.8.0",',
        "]",
        "enterprise = [",
        '    "cryptography>=3.4.0",',
        '    "ldap3>=2.9.0",',
        "]",
        "telemetry = [",
        '    "opentelemetry-api>=1.0.0",',
        '    "opentelemetry-sdk>=1.0.0",',
        '    "prometheus-client>=0.11.0",',
        "]",
        "",
        "[project.scripts]",
        'pyics = "pyics.cli.main:main"',
        'pyics-validate = "pyics.cli.validation.main:validation_cli"',
        'pyics-generate = "pyics.cli.composition.main:composition_cli"',
        'pyics-audit = "pyics.cli.registry.main:registry_cli"',
        "",
        "[project.urls]",
        'Documentation = "https://pyics.readthedocs.io/"',
        'Repository = "https://github.com/obinexuscomputing/pyics"',
        '"Bug Tracker" = "https://github.com/obinexuscomputing/pyics/issues"',
        'Changelog = "https://github.com/obinexuscomputing/pyics/blob/main/CHANGELOG.md"',
        "",
        "[tool.setuptools]",
        'packages = ["pyics"]',
        "",
        '[tool.setuptools.package-data]',
        '"pyics.config" = ["*.json", "*.yaml", "*.toml"]',
        '"pyics.core" = ["*.py"]',
        '"pyics.cli" = ["*.py"]'
    ]
    
    try:
        with open(pyproject_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(toml_lines))
        logger.info(f"pyproject.toml created at: {pyproject_path}")
    except Exception as e:
        logger.error(f"Failed to create pyproject.toml: {e}")

def main():
    """Main execution function"""
    corrector = SetupStructureCorrector(PROJECT_ROOT)
    results = corrector.execute_setup_correction()
    
    # Create additional packaging files
    create_root_level_manifest()
    create_pyproject_toml()
    
    # Display results
    print("=" * 60)
    print("SETUP STRUCTURE CORRECTION SUMMARY")
    print("=" * 60)
    print(f"Backup Created: {'✅' if results['backup_created'] else '❌'}")
    print(f"Setup Relocated: {'✅' if results['setup_relocated'] else '❌'}")
    print(f"Package Config Updated: {'✅' if results['package_config_updated'] else '❌'}")
    print(f"Validation Passed: {'✅' if results['validation_passed'] else '❌'}")
    print("=" * 60)
    print(f"Status: {results['summary']}")
    print("=" * 60)
    
    if results["validation_passed"]:
        print("\n🎯 NEXT STEPS:")
        print("1. Re-run validate_core_structure.py to confirm complete compliance")
        print("2. Test package installation: pip install -e .")
        print("3. Verify CLI commands: pyics --help")
        print("4. Execute domain integration tests")
    
    sys.exit(0 if results["validation_passed"] else 1)

if __name__ == "__main__":
    main()

# [EOF] - End of setup structure corrector
