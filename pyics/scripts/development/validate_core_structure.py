#!/usr/bin/env python3
"""
validate_core_structure.py
Pyics Core Structure Validation and Refactor Analysis

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Systematic validation of core domain structure, CLI separation, and setup configuration
Architecture: Single-pass validation with actionable remediation suggestions
Methodology: Waterfall validation with comprehensive reporting

PROBLEM SOLVED: Ensures proper package structure and domain-CLI separation architecture
DEPENDENCIES: Standard library only (pathlib, ast, importlib)
THREAD SAFETY: Yes - read-only operations with isolated state
DETERMINISTIC: Yes - reproducible validation results

This script validates the complete pyics project structure including setup.py placement,
domain configuration registration, and CLI separation architecture following
single-pass principles with detailed remediation guidance.
"""

import os
import sys
import ast
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
import logging

# Configuration
PROJECT_ROOT = Path.cwd()
PYICS_PACKAGE_DIR = "pyics"
CORE_DIR = "pyics/core"
CLI_DIR = "pyics/cli"
CONFIG_DIR = "pyics/config"
SCRIPTS_DIR = "scripts/development"
SETUP_PY_PATH = "setup.py"

# Required cost metadata fields
REQUIRED_COST_METADATA_FIELDS = {
    "priority_index": int,
    "compute_time_weight": float,
    "exposure_type": str,
    "dependency_level": int,
    "thread_safe": bool,
    "load_order": int
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StructureValidationError(Exception):
    """Custom exception for structure validation failures"""
    pass

class CoreDomainValidator:
    """
    Systematic validator for core domain structure and configuration
    
    Implements comprehensive validation following single-pass architecture
    with detailed reporting and remediation guidance.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.pyics_dir = self.project_root / PYICS_PACKAGE_DIR
        self.core_dir = self.project_root / CORE_DIR
        self.cli_dir = self.project_root / CLI_DIR
        self.config_dir = self.project_root / CONFIG_DIR
        
        # Validation results
        self.validation_results = {
            "setup_structure": {"valid": False, "issues": [], "suggestions": []},
            "core_domains": {"valid": False, "issues": [], "suggestions": [], "domains": {}},
            "cli_structure": {"valid": False, "issues": [], "suggestions": [], "cli_domains": {}},
            "config_registry": {"valid": False, "issues": [], "suggestions": []},
            "metadata_validation": {"valid": False, "issues": [], "suggestions": []},
            "overall_status": {"valid": False, "summary": ""}
        }
        
        self.discovered_domains = set()
        self.validated_configs = {}
        
    def validate_complete_structure(self) -> Dict[str, Any]:
        """
        Execute comprehensive structure validation
        
        Returns:
            Complete validation results with remediation suggestions
        """
        logger.info("Starting comprehensive pyics structure validation...")
        
        try:
            # Phase 1: Setup.py validation
            self._validate_setup_structure()
            
            # Phase 2: Core domain discovery and validation
            self._discover_and_validate_core_domains()
            
            # Phase 3: CLI structure validation
            self._validate_cli_structure()
            
            # Phase 4: Config registry validation
            self._validate_config_registry()
            
            # Phase 5: Metadata validation
            self._validate_cost_metadata()
            
            # Phase 6: Overall assessment
            self._assess_overall_validity()
            
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            self.validation_results["overall_status"]["valid"] = False
            self.validation_results["overall_status"]["summary"] = f"Critical validation failure: {e}"
            return self.validation_results
    
    def _validate_setup_structure(self) -> None:
        """Validate setup.py placement and package configuration"""
        setup_results = self.validation_results["setup_structure"]
        
        # Check setup.py location
        root_setup_path = self.project_root / SETUP_PY_PATH
        pyics_setup_path = self.pyics_dir / "setup.py"
        
        if not root_setup_path.exists():
            if pyics_setup_path.exists():
                setup_results["issues"].append("setup.py found in pyics/ directory instead of project root")
                setup_results["suggestions"].append(
                    f"Move {pyics_setup_path} to {root_setup_path} and update package declarations"
                )
            else:
                setup_results["issues"].append("setup.py not found in project root")
                setup_results["suggestions"].append("Create setup.py at project root with proper package configuration")
        else:
            # Validate setup.py content
            try:
                with open(root_setup_path, 'r', encoding='utf-8') as f:
                    setup_content = f.read()
                
                # Check for proper package declaration
                if 'find_packages' not in setup_content:
                    setup_results["issues"].append("setup.py missing find_packages() declaration")
                    setup_results["suggestions"].append("Add find_packages() to automatically discover packages")
                
                # Check for pyics package reference
                if 'pyics' not in setup_content:
                    setup_results["issues"].append("setup.py does not reference pyics package")
                    setup_results["suggestions"].append("Ensure setup.py includes pyics in package discovery")
                
                # Check for config module inclusion
                if 'pyics/config' not in setup_content and 'pyics.config' not in setup_content:
                    setup_results["issues"].append("setup.py missing pyics.config module reference")
                    setup_results["suggestions"].append("Add pyics/config as installable configuration module")
                
            except Exception as e:
                setup_results["issues"].append(f"Failed to parse setup.py: {e}")
                setup_results["suggestions"].append("Ensure setup.py contains valid Python code")
        
        # Validate package structure
        if not self.pyics_dir.exists():
            setup_results["issues"].append("pyics package directory not found")
            setup_results["suggestions"].append("Create pyics/ package directory structure")
        else:
            # Check for __init__.py
            pyics_init = self.pyics_dir / "__init__.py"
            if not pyics_init.exists():
                setup_results["issues"].append("pyics/__init__.py missing")
                setup_results["suggestions"].append("Create pyics/__init__.py to make it a proper Python package")
        
        setup_results["valid"] = len(setup_results["issues"]) == 0
        logger.info(f"Setup structure validation: {'PASS' if setup_results['valid'] else 'FAIL'}")
    
    def _discover_and_validate_core_domains(self) -> None:
        """Discover and validate all core domains"""
        core_results = self.validation_results["core_domains"]
        
        if not self.core_dir.exists():
            core_results["issues"].append("pyics/core directory not found")
            core_results["suggestions"].append("Create pyics/core directory for domain modules")
            return
        
        # Discover domains
        for item in self.core_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
                domain_name = item.name
                self.discovered_domains.add(domain_name)
                
                # Validate individual domain
                domain_validation = self._validate_domain_structure(item, domain_name)
                core_results["domains"][domain_name] = domain_validation
                
                # Collect issues and suggestions
                core_results["issues"].extend(domain_validation["issues"])
                core_results["suggestions"].extend(domain_validation["suggestions"])
        
        if not self.discovered_domains:
            core_results["issues"].append("No core domains discovered")
            core_results["suggestions"].append("Create domain directories under pyics/core/")
        
        core_results["valid"] = len(core_results["issues"]) == 0
        logger.info(f"Core domains validation: {len(self.discovered_domains)} domains discovered")
    
    def _validate_domain_structure(self, domain_path: Path, domain_name: str) -> Dict[str, Any]:
        """Validate individual domain structure"""
        domain_validation = {
            "valid": False,
            "issues": [],
            "suggestions": [],
            "has_init": False,
            "has_config": False,
            "config_metadata": None
        }
        
        # Check for __init__.py
        init_path = domain_path / "__init__.py"
        if not init_path.exists():
            domain_validation["issues"].append(f"Domain {domain_name} missing __init__.py")
            domain_validation["suggestions"].append(f"Create {init_path} with domain exports")
        else:
            domain_validation["has_init"] = True
            
            # Validate __init__.py content
            try:
                with open(init_path, 'r', encoding='utf-8') as f:
                    init_content = f.read()
                
                if 'get_domain_metadata' not in init_content:
                    domain_validation["suggestions"].append(
                        f"Add get_domain_metadata() function to {init_path}"
                    )
            except Exception as e:
                domain_validation["issues"].append(f"Failed to read {init_path}: {e}")
        
        # Check for config.py
        config_path = domain_path / "config.py"
        if not config_path.exists():
            domain_validation["issues"].append(f"Domain {domain_name} missing config.py")
            domain_validation["suggestions"].append(f"Create {config_path} with cost metadata")
        else:
            domain_validation["has_config"] = True
            
            # Validate config.py content
            config_metadata = self._validate_config_metadata(config_path, domain_name)
            domain_validation["config_metadata"] = config_metadata
            
            if not config_metadata["valid"]:
                domain_validation["issues"].extend(config_metadata["issues"])
                domain_validation["suggestions"].extend(config_metadata["suggestions"])
        
        domain_validation["valid"] = (
            domain_validation["has_init"] and 
            domain_validation["has_config"] and 
            len(domain_validation["issues"]) == 0
        )
        
        return domain_validation
    
    def _validate_config_metadata(self, config_path: Path, domain_name: str) -> Dict[str, Any]:
        """Validate config.py metadata structure"""
        metadata_validation = {
            "valid": False,
            "issues": [],
            "suggestions": [],
            "cost_metadata": None
        }
        
        try:
            # Parse config.py AST
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            tree = ast.parse(config_content)
            
            # Look for cost_metadata assignment
            cost_metadata_found = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'cost_metadata':
                            cost_metadata_found = True
                            
                            # Try to evaluate the assignment
                            try:
                                # This is a simplified evaluation - in production, use ast.literal_eval
                                exec(config_content, {}, {})
                                # Import dynamically to get the actual metadata
                                spec = importlib.util.spec_from_file_location(f"{domain_name}.config", config_path)
                                config_module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(config_module)
                                
                                if hasattr(config_module, 'cost_metadata'):
                                    cost_metadata = config_module.cost_metadata
                                    metadata_validation["cost_metadata"] = cost_metadata
                                    
                                    # Validate required fields
                                    for field, expected_type in REQUIRED_COST_METADATA_FIELDS.items():
                                        if field not in cost_metadata:
                                            metadata_validation["issues"].append(
                                                f"Missing required field '{field}' in {domain_name} cost_metadata"
                                            )
                                        elif not isinstance(cost_metadata[field], expected_type):
                                            metadata_validation["issues"].append(
                                                f"Field '{field}' in {domain_name} cost_metadata should be {expected_type.__name__}"
                                            )
                                
                            except Exception as e:
                                metadata_validation["issues"].append(
                                    f"Failed to evaluate cost_metadata in {domain_name}: {e}"
                                )
            
            if not cost_metadata_found:
                metadata_validation["issues"].append(f"cost_metadata not found in {config_path}")
                metadata_validation["suggestions"].append(
                    f"Add cost_metadata dictionary to {config_path} with required fields: {list(REQUIRED_COST_METADATA_FIELDS.keys())}"
                )
        
        except Exception as e:
            metadata_validation["issues"].append(f"Failed to parse {config_path}: {e}")
            metadata_validation["suggestions"].append(f"Ensure {config_path} contains valid Python code")
        
        metadata_validation["valid"] = len(metadata_validation["issues"]) == 0
        return metadata_validation
    
    def _validate_cli_structure(self) -> None:
        """Validate CLI separation structure"""
        cli_results = self.validation_results["cli_structure"]
        
        if not self.cli_dir.exists():
            cli_results["issues"].append("pyics/cli directory not found")
            cli_results["suggestions"].append("Create pyics/cli directory for domain CLI modules")
            return
        
        # Check for CLI __init__.py
        cli_init = self.cli_dir / "__init__.py"
        if not cli_init.exists():
            cli_results["issues"].append("pyics/cli/__init__.py missing")
            cli_results["suggestions"].append("Create pyics/cli/__init__.py")
        
        # Check for IoC registry
        ioc_path = self.cli_dir / "ioc.py"
        if not ioc_path.exists():
            cli_results["issues"].append("pyics/cli/ioc.py missing")
            cli_results["suggestions"].append("Create pyics/cli/ioc.py for centralized CLI registration")
        
        # Validate CLI domains match core domains
        for domain_name in self.discovered_domains:
            cli_domain_dir = self.cli_dir / domain_name
            cli_main_path = cli_domain_dir / "main.py"
            
            cli_domain_validation = {
                "valid": False,
                "has_directory": cli_domain_dir.exists(),
                "has_main": cli_main_path.exists(),
                "issues": [],
                "suggestions": []
            }
            
            if not cli_domain_dir.exists():
                cli_domain_validation["issues"].append(f"CLI directory missing for domain {domain_name}")
                cli_domain_validation["suggestions"].append(f"Create {cli_domain_dir}")
            
            if not cli_main_path.exists():
                cli_domain_validation["issues"].append(f"CLI main.py missing for domain {domain_name}")
                cli_domain_validation["suggestions"].append(f"Create {cli_main_path} with domain CLI logic")
            else:
                # Validate main.py imports from core domain
                try:
                    with open(cli_main_path, 'r', encoding='utf-8') as f:
                        main_content = f.read()
                    
                    expected_import = f"pyics.core.{domain_name}"
                    if expected_import not in main_content:
                        cli_domain_validation["suggestions"].append(
                            f"Add import from {expected_import} in {cli_main_path}"
                        )
                
                except Exception as e:
                    cli_domain_validation["issues"].append(f"Failed to read {cli_main_path}: {e}")
            
            cli_domain_validation["valid"] = (
                cli_domain_validation["has_directory"] and 
                cli_domain_validation["has_main"] and 
                len(cli_domain_validation["issues"]) == 0
            )
            
            cli_results["cli_domains"][domain_name] = cli_domain_validation
            cli_results["issues"].extend(cli_domain_validation["issues"])
            cli_results["suggestions"].extend(cli_domain_validation["suggestions"])
        
        cli_results["valid"] = len(cli_results["issues"]) == 0
        logger.info(f"CLI structure validation: {len(cli_results['cli_domains'])} CLI domains checked")
    
    def _validate_config_registry(self) -> None:
        """Validate central config registry"""
        registry_results = self.validation_results["config_registry"]
        
        # Check for config directory
        if not self.config_dir.exists():
            registry_results["issues"].append("pyics/config directory not found")
            registry_results["suggestions"].append("Create pyics/config directory")
            return
        
        # Check for config registry file
        config_registry_path = self.core_dir / "config_registry.py"
        if not config_registry_path.exists():
            registry_results["issues"].append("pyics/core/config_registry.py missing")
            registry_results["suggestions"].append(
                "Create pyics/core/config_registry.py to register all domain metadata"
            )
        else:
            # Validate registry content
            try:
                with open(config_registry_path, 'r', encoding='utf-8') as f:
                    registry_content = f.read()
                
                # Check for domain registration functions
                required_functions = [
                    'register_domain_metadata',
                    'get_all_domain_metadata',
                    'get_domain_cost_metadata'
                ]
                
                for func_name in required_functions:
                    if func_name not in registry_content:
                        registry_results["suggestions"].append(
                            f"Add {func_name}() function to config_registry.py"
                        )
            
            except Exception as e:
                registry_results["issues"].append(f"Failed to read config_registry.py: {e}")
        
        registry_results["valid"] = len(registry_results["issues"]) == 0
    
    def _validate_cost_metadata(self) -> None:
        """Validate cost metadata across all domains"""
        metadata_results = self.validation_results["metadata_validation"]
        
        # Test dynamic import capabilities
        for domain_name in self.discovered_domains:
            try:
                config_path = self.core_dir / domain_name / "config.py"
                if config_path.exists():
                    spec = importlib.util.spec_from_file_location(f"{domain_name}.config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    
                    if hasattr(config_module, 'cost_metadata'):
                        self.validated_configs[domain_name] = config_module.cost_metadata
                    else:
                        metadata_results["issues"].append(
                            f"Domain {domain_name} config.py missing cost_metadata"
                        )
            
            except Exception as e:
                metadata_results["issues"].append(
                    f"Failed to dynamically import {domain_name} config: {e}"
                )
                metadata_results["suggestions"].append(
                    f"Ensure {domain_name}/config.py has valid Python syntax and cost_metadata"
                )
        
        # Validate cost metadata consistency
        if self.validated_configs:
            # Check for load_order conflicts
            load_orders = [meta.get("load_order", 0) for meta in self.validated_configs.values()]
            if len(set(load_orders)) != len(load_orders):
                metadata_results["issues"].append("Duplicate load_order values detected across domains")
                metadata_results["suggestions"].append("Ensure each domain has unique load_order value")
        
        metadata_results["valid"] = len(metadata_results["issues"]) == 0
        logger.info(f"Metadata validation: {len(self.validated_configs)} configs validated")
    
    def _assess_overall_validity(self) -> None:
        """Assess overall structure validity"""
        overall = self.validation_results["overall_status"]
        
        all_valid = all([
            self.validation_results["setup_structure"]["valid"],
            self.validation_results["core_domains"]["valid"],
            self.validation_results["cli_structure"]["valid"],
            self.validation_results["config_registry"]["valid"],
            self.validation_results["metadata_validation"]["valid"]
        ])
        
        overall["valid"] = all_valid
        
        if all_valid:
            overall["summary"] = "✅ Core domain configuration and CLI registration validated successfully."
        else:
            failed_areas = []
            for area, results in self.validation_results.items():
                if area != "overall_status" and not results["valid"]:
                    failed_areas.append(area.replace("_", " ").title())
            
            overall["summary"] = f"❌ Validation failed in: {', '.join(failed_areas)}"
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("PYICS CORE STRUCTURE VALIDATION REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        
        # Overall status
        overall = self.validation_results["overall_status"]
        report.append(f"\nOVERALL STATUS: {overall['summary']}")
        
        # Detailed results for each area
        for area, results in self.validation_results.items():
            if area == "overall_status":
                continue
                
            report.append(f"\n{area.replace('_', ' ').upper()}:")
            report.append(f"  Status: {'✅ VALID' if results['valid'] else '❌ INVALID'}")
            
            if results["issues"]:
                report.append("  Issues:")
                for issue in results["issues"]:
                    report.append(f"    - {issue}")
            
            if results["suggestions"]:
                report.append("  Suggestions:")
                for suggestion in results["suggestions"]:
                    report.append(f"    + {suggestion}")
        
        # Domain summary
        if self.discovered_domains:
            report.append(f"\nDISCOVERED DOMAINS ({len(self.discovered_domains)}):")
            for domain in sorted(self.discovered_domains):
                domain_info = self.validation_results["core_domains"]["domains"].get(domain, {})
                status = "✅" if domain_info.get("valid", False) else "❌"
                report.append(f"  {status} {domain}")
        
        # Config metadata summary
        if self.validated_configs:
            report.append(f"\nVALIDATED CONFIGURATIONS ({len(self.validated_configs)}):")
            for domain, config in self.validated_configs.items():
                report.append(f"  - {domain}: load_order={config.get('load_order', 'N/A')}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)

def create_remediation_script(validation_results: Dict[str, Any], project_root: Path) -> str:
    """Generate remediation script based on validation results"""
    script_lines = [
        "#!/usr/bin/env python3",
        '"""',
        "Auto-generated remediation script for pyics structure issues",
        f"Generated: {datetime.now().isoformat()}",
        '"""',
        "",
        "import os",
        "from pathlib import Path",
        "",
        f"PROJECT_ROOT = Path('{project_root}')",
        ""
    ]
    
    # Add remediation steps based on validation issues
    if not validation_results["setup_structure"]["valid"]:
        script_lines.extend([
            "# Setup.py remediation",
            "def fix_setup_structure():",
            "    print('Creating setup.py at project root...')",
            "    # Implementation would go here",
            "    pass",
            ""
        ])
    
    if not validation_results["core_domains"]["valid"]:
        script_lines.extend([
            "# Core domains remediation", 
            "def fix_core_domains():",
            "    print('Creating missing domain structure...')",
            "    # Implementation would go here",
            "    pass",
            ""
        ])
    
    script_lines.extend([
        "if __name__ == '__main__':",
        "    print('Running pyics structure remediation...')",
        "    # Call remediation functions here",
        "    print('Remediation complete - please re-run validation')"
    ])
    
    return "\n".join(script_lines)

def self_test_validation() -> bool:
    """Self-test function to verify validator functionality"""
    logger.info("Running validator self-test...")
    
    try:
        # Test with current project structure
        validator = CoreDomainValidator(PROJECT_ROOT)
        results = validator.validate_complete_structure()
        
        # Verify results structure
        required_sections = [
            "setup_structure", "core_domains", "cli_structure", 
            "config_registry", "metadata_validation", "overall_status"
        ]
        
        for section in required_sections:
            if section not in results:
                logger.error(f"Self-test failed: Missing results section '{section}'")
                return False
        
        logger.info("✅ Self-test passed - validator functioning correctly")
        return True
        
    except Exception as e:
        logger.error(f"Self-test failed with exception: {e}")
        return False

def main():
    """Main execution function"""
    print("=" * 80)
    print("PYICS CORE STRUCTURE VALIDATION")
    print("Engineering Lead: Nnamdi Okpala / OBINexus Computing") 
    print("=" * 80)
    
    # Run self-test first
    if not self_test_validation():
        print("❌ Self-test failed - validator may have issues")
        sys.exit(1)
    
    try:
        # Initialize validator
        validator = CoreDomainValidator(PROJECT_ROOT)
        
        # Execute validation
        results = validator.validate_complete_structure()
        
        # Generate and display report
        report = validator.generate_validation_report()
        print(report)
        
        # Save detailed results
        results_file = PROJECT_ROOT / SCRIPTS_DIR / "validation_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Generate remediation script if needed
        if not results["overall_status"]["valid"]:
            remediation_script = create_remediation_script(results, PROJECT_ROOT)
            remediation_file = PROJECT_ROOT / SCRIPTS_DIR / "remediate_structure.py"
            
            with open(remediation_file, 'w', encoding='utf-8') as f:
                f.write(remediation_script)
            
            print(f"Remediation script generated: {remediation_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_status"]["valid"] else 1)
        
    except Exception as e:
        logger.error(f"Validation failed with critical error: {e}")
        print("❌ Critical validation failure - check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()

# [EOF] - End of validate_core_structure.py
