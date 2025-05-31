#!/usr/bin/env python3
"""
scripts/generate_setup.py
Pyics Setup.py Generator with Single-Pass Architecture Support

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Generate setup.py with proper first-pass module exposure
Architecture: Single-pass dependency-safe package configuration
Phase: 3.1.6.2 - Setup.py Generation and Validation

PROBLEM SOLVED: Creates setup.py with correct package discovery and metadata
DEPENDENCIES: Standard library, existing domain structure
THREAD SAFETY: Yes - atomic file operations
DETERMINISTIC: Yes - consistent setup.py generation

This script generates a proper setup.py file that only exposes first-pass
modules and maintains single-pass architecture compliance.
"""

import os
import sys
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CORE_DIR = PROJECT_ROOT / "pyics" / "core"
CLI_DIR = PROJECT_ROOT / "pyics" / "cli"

# First-pass modules (single-pass architecture compliance)
FIRST_PASS_DOMAINS = {
    "primitives": {
        "load_order": 10,
        "priority_index": 1,
        "description": "Atomic operations and thread-safe building blocks",
        "dependencies": []
    },
    "protocols": {
        "load_order": 20,
        "priority_index": 1,
        "description": "Type safety contracts and interface definitions",
        "dependencies": []
    },
    "structures": {
        "load_order": 30,
        "priority_index": 2,
        "description": "Immutable data containers for calendar operations",
        "dependencies": ["primitives", "protocols"]
    }
}

class SetupGenerator:
    """
    Generates setup.py with single-pass architecture compliance
    
    Ensures only first-pass modules are exposed and maintains
    dependency safety throughout the package structure.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.core_dir = self.project_root / "pyics" / "core"
        self.cli_dir = self.project_root / "pyics" / "cli"
        self.discovered_domains = []
        self.package_structure = {}
        
    def discover_domains(self) -> List[str]:
        """Discover available first-pass domains in core directory"""
        logger.info("Discovering first-pass domains...")
        
        discovered = []
        
        if not self.core_dir.exists():
            logger.warning(f"Core directory not found: {self.core_dir}")
            return discovered
        
        for item in self.core_dir.iterdir():
            if item.is_dir() and item.name in FIRST_PASS_DOMAINS:
                # Validate domain structure
                if self._validate_domain_structure(item):
                    discovered.append(item.name)
                    logger.info(f"✅ Discovered first-pass domain: {item.name}")
                else:
                    logger.warning(f"⚠️  Domain structure incomplete: {item.name}")
        
        # Sort by load order to maintain single-pass compliance
        discovered.sort(key=lambda x: FIRST_PASS_DOMAINS[x]["load_order"])
        self.discovered_domains = discovered
        
        logger.info(f"First-pass domains discovered: {', '.join(discovered) if discovered else 'None'}")
        return discovered
    
    def _validate_domain_structure(self, domain_path: Path) -> bool:
        """Validate that domain has required structure for first-pass exposure"""
        required_files = ["__init__.py", "config.py"]
        
        for required_file in required_files:
            if not (domain_path / required_file).exists():
                logger.warning(f"Missing required file: {domain_path.name}/{required_file}")
                return False
        
        return True
    
    def analyze_package_structure(self) -> Dict[str, Any]:
        """Analyze complete package structure for setup.py generation"""
        logger.info("Analyzing package structure...")
        
        structure = {
            "core_domains": self.discovered_domains,
            "cli_available": self.cli_dir.exists(),
            "package_data": {},
            "entry_points": {},
            "dependencies": self._analyze_dependencies()
        }
        
        # Analyze package data for each domain
        for domain in self.discovered_domains:
            domain_path = self.core_dir / domain
            domain_files = []
            
            for file_path in domain_path.rglob("*.py"):
                relative_path = file_path.relative_to(domain_path)
                domain_files.append(str(relative_path))
            
            # Include documentation files
            for file_path in domain_path.rglob("*.md"):
                relative_path = file_path.relative_to(domain_path)
                domain_files.append(str(relative_path))
            
            structure["package_data"][f"pyics.core.{domain}"] = domain_files
        
        # Analyze CLI structure
        if structure["cli_available"]:
            cli_files = []
            for file_path in self.cli_dir.rglob("*.py"):
                relative_path = file_path.relative_to(self.cli_dir)
                cli_files.append(str(relative_path))
            
            structure["package_data"]["pyics.cli"] = cli_files
            
            # Generate CLI entry points
            structure["entry_points"]["console_scripts"] = [
                "pyics=pyics.cli.main:main"
            ]
            
            # Add domain-specific entry points
            for domain in self.discovered_domains:
                structure["entry_points"]["console_scripts"].append(
                    f"pyics-{domain}=pyics.cli.main:{domain}"
                )
        
        self.package_structure = structure
        logger.info("Package structure analysis complete")
        return structure
    
    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze inter-domain dependencies for validation"""
        dependencies = {}
        
        for domain in self.discovered_domains:
            domain_deps = FIRST_PASS_DOMAINS[domain]["dependencies"]
            dependencies[domain] = domain_deps
            
            # Validate dependencies are also first-pass
            for dep in domain_deps:
                if dep not in FIRST_PASS_DOMAINS:
                    logger.warning(f"Domain {domain} depends on non-first-pass domain: {dep}")
        
        return dependencies
    
    def get_package_metadata(self) -> Dict[str, Any]:
        """Get comprehensive package metadata for setup.py"""
        # Try to read existing metadata from pyics/__init__.py
        version = self._extract_version()
        
        return {
            "name": "pyics",
            "version": version,
            "description": "Pyics - Data-Oriented Calendar Automation System with Single-Pass Architecture",
            "long_description_content_type": "text/markdown",
            "author": "Nnamdi Okpala / OBINexus Computing",
            "author_email": "engineering@obinexus.com",
            "url": "https://github.com/obinexuscomputing/pyics",
            "license": "MIT",
            "python_requires": ">=3.8",
            "classifiers": [
                "Development Status :: 4 - Beta",
                "Intended Audience :: Developers",
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
            "keywords": [
                "calendar", "automation", "scheduling", "icalendar", "ics",
                "data-oriented-programming", "domain-driven-design", "enterprise",
                "single-pass-architecture", "dependency-injection"
            ]
        }
    
    def _extract_version(self) -> str:
        """Extract version from pyics/__init__.py or use default"""
        version = "1.0.0"  # Default version
        
        init_path = self.project_root / "pyics" / "__init__.py"
        if init_path.exists():
            try:
                with open(init_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find __version__
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "__version__":
                                if isinstance(node.value, ast.Constant):
                                    version = node.value.value
                                elif isinstance(node.value, ast.Str):  # Python < 3.8
                                    version = node.value.s
                                break
            except Exception as e:
                logger.warning(f"Could not extract version from __init__.py: {e}")
        
        logger.info(f"Package version: {version}")
        return version
    
    def generate_setup_content(self) -> str:
        """Generate complete setup.py content with single-pass compliance"""
        logger.info("Generating setup.py content...")
        
        metadata = self.get_package_metadata()
        structure = self.package_structure
        timestamp = datetime.now().isoformat()
        
        # Read README if available
        long_description = self._get_long_description(metadata["description"])
        
        # Generate package data section
        package_data_lines = []
        for package, files in structure["package_data"].items():
            if files:
                package_data_lines.append(f"        '{package}': ['*.py', '*.md'],")
        
        # Generate entry points section
        entry_points_lines = []
        if "console_scripts" in structure["entry_points"]:
            for script in structure["entry_points"]["console_scripts"]:
                entry_points_lines.append(f"            '{script}',")
        
        # Generate domain list for documentation
        domain_list = ", ".join(self.discovered_domains) if self.discovered_domains else "None discovered"
        
        setup_content = f'''#!/usr/bin/env python3
"""
setup.py
Pyics Package Configuration - Single-Pass Architecture

Generated: {timestamp}
Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Python package configuration with first-pass module exposure
Architecture: Single-pass dependency-safe package structure

PROBLEM SOLVED: Establishes single-pass package structure with dependency safety
DEPENDENCIES: setuptools, find_packages for automatic package discovery
PACKAGE STRUCTURE: pyics/ as installable package with first-pass domains only
FIRST-PASS DOMAINS: {domain_list}

This setup.py exposes only first-pass modules following single-pass architecture
principles to prevent circular dependencies and ensure deterministic loading.

DOMAIN LOAD ORDER:
{chr(10).join([f"  - {domain}: load_order {FIRST_PASS_DOMAINS[domain]['load_order']}" for domain in self.discovered_domains])}
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    raise RuntimeError("Pyics requires Python 3.8 or higher")

# Read README for long description
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    try:
        with open(readme_path, encoding='utf-8') as f:
            long_description = f.read()
    except Exception:
        long_description = "{metadata['description']}"
else:
    long_description = "{metadata['description']}"

# Core dependencies (minimal for single-pass architecture)
CORE_DEPENDENCIES = [
    'click>=8.0.0',              # CLI framework
    'python-dateutil>=2.8.0',   # Date/time utilities
    'typing-extensions>=4.0.0',  # Type hints backport
]

# Optional dependencies grouped by functionality
EXTRAS_REQUIRE = {{
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'black>=21.0.0',
        'mypy>=0.910',
        'flake8>=3.8.0',
        'isort>=5.0.0',
    ],
    'calendar': [
        'icalendar>=4.0.0',
        'pytz>=2021.1',
    ],
    'enterprise': [
        'cryptography>=3.4.0',
        'ldap3>=2.9.0',
    ],
    'telemetry': [
        'opentelemetry-api>=1.0.0',
        'opentelemetry-sdk>=1.0.0',
        'prometheus-client>=0.11.0',
    ],
    'validation': [
        'pydantic>=1.8.0',
        'jsonschema>=3.2.0',
    ],
}}

# Add 'all' extras that includes everything
EXTRAS_REQUIRE['all'] = [
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
]

# Package metadata
setup(
    name="{metadata['name']}",
    version="{metadata['version']}",
    description="{metadata['description']}",
    long_description=long_description,
    long_description_content_type="{metadata['long_description_content_type']}",
    author="{metadata['author']}",
    author_email="{metadata['author_email']}",
    url="{metadata['url']}",
    license="{metadata['license']}",
    
    # Package discovery configuration (single-pass compliance)
    packages=find_packages(include=['pyics', 'pyics.*']),
    package_dir={{'': '.'}},
    
    # Include package data for first-pass domains
    package_data={{
        'pyics': ['*.py', '*.md'],
        'pyics.core': ['*.py', '*.md'],
{chr(10).join(package_data_lines)}
    }},
    include_package_data=True,
    
    # Python version requirements
    python_requires='{metadata["python_requires"]}',
    
    # Dependencies
    install_requires=CORE_DEPENDENCIES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points for CLI commands (first-pass domains only)
    entry_points={{
        'console_scripts': [
{chr(10).join(entry_points_lines)}
        ],
    }},
    
    # Classification metadata
    classifiers={metadata["classifiers"]},
    
    # Keywords for package discovery
    keywords={metadata["keywords"]},
    
    # Project URLs
    project_urls={{
        'Documentation': 'https://pyics.readthedocs.io/',
        'Source': 'https://github.com/obinexuscomputing/pyics',
        'Tracker': 'https://github.com/obinexuscomputing/pyics/issues',
        'Changelog': 'https://github.com/obinexuscomputing/pyics/blob/main/CHANGELOG.md',
        'Funding': 'https://github.com/sponsors/obinexuscomputing',
    }},
    
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
    
    first_pass_domains = {self.discovered_domains}
    validation_passed = True
    
    print("🔍 Validating first-pass domain structure...")
    
    for domain in first_pass_domains:
        domain_path = Path(__file__).parent / "pyics" / "core" / domain
        
        # Check if domain directory exists
        if not domain_path.exists():
            print(f"⚠️  Warning: First-pass domain directory missing: {{domain}}")
            validation_passed = False
            continue
        
        # Check for required files
        required_files = ["__init__.py", "config.py"]
        for required_file in required_files:
            if not (domain_path / required_file).exists():
                print(f"⚠️  Warning: Missing required file: {{domain}}/{{required_file}}")
                validation_passed = False
        
        # Try to import domain
        try:
            spec = importlib.util.find_spec(f"pyics.core.{{domain}}")
            if spec is None:
                print(f"⚠️  Warning: First-pass domain '{{domain}}' not importable")
                validation_passed = False
            else:
                print(f"✅ First-pass domain validated: {{domain}}")
        except ImportError as e:
            print(f"⚠️  Warning: Cannot validate first-pass domain '{{domain}}': {{e}}")
            validation_passed = False
    
    if validation_passed:
        print(f"✅ First-pass domain validation passed: {{', '.join(first_pass_domains)}}")
        print(f"📊 Architecture: Single-pass compliant with {{len(first_pass_domains)}} domains")
    else:
        print("⚠️  First-pass domain validation completed with warnings")
    
    # Display domain load order
    print("📋 Domain load order:")
    for domain in sorted(first_pass_domains, key=lambda x: FIRST_PASS_DOMAINS.get(x, {{}}).get("load_order", 999)):
        load_order = FIRST_PASS_DOMAINS.get(domain, {{}}).get("load_order", "Unknown")
        description = FIRST_PASS_DOMAINS.get(domain, {{}}).get("description", "No description")
        print(f"  {{load_order:2d}}. {{domain}} - {{description}}")
    
    return validation_passed

# Domain dependency validation
def validate_dependencies():
    """Validate domain dependencies follow single-pass architecture"""
    first_pass_domains = {self.discovered_domains}
    dependency_violations = []
    
    print("🔗 Validating dependency architecture...")
    
    for domain in first_pass_domains:
        domain_spec = FIRST_PASS_DOMAINS.get(domain, {{}})
        dependencies = domain_spec.get("dependencies", [])
        current_load_order = domain_spec.get("load_order", 999)
        
        for dep in dependencies:
            if dep not in FIRST_PASS_DOMAINS:
                dependency_violations.append(f"{{domain}} depends on non-first-pass domain: {{dep}}")
                continue
            
            dep_load_order = FIRST_PASS_DOMAINS[dep]["load_order"]
            if dep_load_order >= current_load_order:
                dependency_violations.append(f"{{domain}} (load_order {{current_load_order}}) depends on {{dep}} (load_order {{dep_load_order}}) - violates single-pass architecture")
    
    if dependency_violations:
        print("❌ Dependency violations detected:")
        for violation in dependency_violations:
            print(f"  - {{violation}}")
        return False
    else:
        print("✅ Dependency architecture validation passed")
        return True

if __name__ == "__main__":
    print("🚀 Pyics Setup.py - Single-Pass Architecture")
    print("=" * 50)
    
    # Run validations
    domain_validation = validate_first_pass_domains()
    dependency_validation = validate_dependencies()
    
    if domain_validation and dependency_validation:
        print("🎯 Setup.py validation completed successfully!")
        print("📦 Package ready for installation with: pip install -e .")
    else:
        print("⚠️  Setup.py validation completed with warnings")
        print("💡 Consider running domain generation scripts to fix issues")
    
    print("=" * 50)

# [EOF] - End of single-pass architecture setup.py
'''
        
        logger.info("Setup.py content generation complete")
        return setup_content
    
    def _get_long_description(self, fallback: str) -> str:
        """Get long description from README or use fallback"""
        readme_path = self.project_root / "README.md"
        
        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Could not read README.md: {e}")
        
        return fallback
    
    def write_setup_py(self, content: str) -> bool:
        """Write setup.py content to file"""
        setup_path = self.project_root / "setup.py"
        
        try:
            with open(setup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Setup.py written successfully: {setup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to write setup.py: {e}")
            return False
    
    def validate_setup_py(self) -> bool:
        """Validate generated setup.py"""
        setup_path = self.project_root / "setup.py"
        
        if not setup_path.exists():
            logger.error("Setup.py was not created")
            return False
        
        try:
            # Check syntax
            with open(setup_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            compile(content, str(setup_path), 'exec')
            logger.info("✅ Setup.py syntax validation passed")
            
            # Check if it can be executed
            exec(content)
            logger.info("✅ Setup.py execution validation passed")
            
            return True
            
        except SyntaxError as e:
            logger.error(f"❌ Setup.py syntax error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Setup.py execution error: {e}")
            return False
    
    def generate_setup_py(self) -> bool:
        """Complete setup.py generation process"""
        logger.info("🚀 Starting setup.py generation process...")
        
        try:
            # Discover domains
            domains = self.discover_domains()
            
            if not domains:
                logger.warning("No first-pass domains discovered. Generating minimal setup.py.")
            
            # Analyze package structure
            self.analyze_package_structure()
            
            # Generate content
            content = self.generate_setup_content()
            
            # Write setup.py
            if not self.write_setup_py(content):
                return False
            
            # Validate setup.py
            if not self.validate_setup_py():
                return False
            
            logger.info("🎯 Setup.py generation completed successfully!")
            
            # Display summary
            self._display_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup.py generation failed: {e}")
            return False
    
    def _display_summary(self):
        """Display generation summary"""
        print("\n" + "=" * 60)
        print("SETUP.PY GENERATION SUMMARY")
        print("=" * 60)
        print(f"📦 Package: pyics")
        print(f"🏗️  Architecture: Single-pass compliant")
        print(f"🎯 First-pass domains: {', '.join(self.discovered_domains) if self.discovered_domains else 'None'}")
        print(f"📁 CLI available: {'✅' if self.package_structure.get('cli_available') else '❌'}")
        print(f"🔧 Entry points: {len(self.package_structure.get('entry_points', {}).get('console_scripts', []))}")
        print("=" * 60)
        print("🚀 Next steps:")
        print("  1. Test installation: pip install -e .")
        print("  2. Verify imports: python -c 'import pyics'")
        print("  3. Test CLI: pyics --help")
        print("=" * 60)

def main():
    """Main execution function"""
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("Usage: python generate_setup.py")
        print("Generates setup.py with single-pass architecture compliance")
        print("Only exposes first-pass domains: primitives, protocols, structures")
        return 0
    
    generator = SetupGenerator(PROJECT_ROOT)
    
    if generator.generate_setup_py():
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())

# [EOF] - End of setup.py generator
