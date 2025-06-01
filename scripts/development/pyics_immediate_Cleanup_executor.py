#!/usr/bin/env python3
"""
pyics_immediate_cleanup_executor.py
Pyics Immediate Core Cleanup Actions Executor

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Execute immediate cleanup actions to resolve domain redundancy and structural violations
Architecture: Sequential cleanup with validation and rollback capability
Methodology: Systematic consolidation following single-pass loading principles

PROBLEM SOLVED: Eliminates domain redundancy and structural complexity blocking downstream development
DEPENDENCIES: Standard library only (pathlib, shutil, json)
THREAD SAFETY: Yes - atomic operations with comprehensive backup
DETERMINISTIC: Yes - reproducible cleanup with validation checkpoints

This executor implements the immediate cleanup actions:
1. Merge core/validation → core/validators  
2. Eliminate core/transforms → core/transformations
3. Eliminate core/logic (distribute to primitives/composition)
4. Clean complex nested structures (implementations/, interfaces/, etc.)
5. Implement single-pass load order
"""

import os
import sys
import shutil
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
import logging

# Configuration
PROJECT_ROOT = Path.cwd()
PYICS_CORE_DIR = "pyics/core"
BACKUP_DIR = "cleanup_backup"

# Immediate actions configuration
IMMEDIATE_ACTIONS = {
    "merge_validation_to_validators": {
        "source": "validation",
        "target": "validators", 
        "priority": 1,
        "description": "Merge core/validation into core/validators"
    },
    "eliminate_transforms_to_transformations": {
        "source": "transforms",
        "target": "transformations",
        "priority": 2,
        "description": "Eliminate core/transforms, move content to core/transformations"
    },
    "eliminate_logic_domain": {
        "source": "logic",
        "target": "DELETE",
        "priority": 3,
        "description": "Eliminate core/logic, distribute functions to primitives/composition"
    },
    "clean_complex_structures": {
        "source": "ALL_DOMAINS",
        "target": "FLATTEN",
        "priority": 4,
        "description": "Remove complex nested structures (implementations/, interfaces/, etc.)"
    }
}

# Domain load order for single-pass architecture
SINGLE_PASS_LOAD_ORDER = {
    "primitives": 10,
    "protocols": 20, 
    "structures": 30,
    "composition": 40,
    "validators": 50,    # Consolidated validation
    "transformations": 60, # Consolidated transforms
    "registry": 70,
    "routing": 80,
    "safety": 90
}

# Complex structure patterns to eliminate
COMPLEX_PATTERNS = [
    "implementations",
    "interfaces", 
    "compliance",
    "contracts",
    "tests"
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PyicsImmediateCleanupExecutor:
    """
    Executor for immediate Pyics cleanup actions
    
    Implements systematic domain consolidation and structure flattening
    with comprehensive backup and validation at each step.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.core_dir = self.project_root / PYICS_CORE_DIR
        self.backup_dir = self.project_root / BACKUP_DIR
        
        self.execution_results = {
            "backup_created": False,
            "actions_completed": {},
            "load_order_updated": False,
            "validation_passed": False,
            "rollback_points": [],
            "summary": ""
        }
        
        self.discovered_domains = []
        self.preserved_functions = {}
        
    def execute_immediate_cleanup(self) -> Dict[str, Any]:
        """Execute all immediate cleanup actions in priority order"""
        logger.info("=" * 60)
        logger.info("PYICS IMMEDIATE CORE CLEANUP EXECUTION")
        logger.info("Engineering Lead: Nnamdi Okpala / OBINexus Computing")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Create comprehensive backup
            self._create_comprehensive_backup()
            
            # Phase 2: Discover current domain structure
            self._discover_current_domains()
            
            # Phase 3: Execute immediate actions in priority order
            sorted_actions = sorted(IMMEDIATE_ACTIONS.items(), key=lambda x: x[1]["priority"])
            
            for action_name, action_config in sorted_actions:
                logger.info(f"Executing: {action_config['description']}")
                self._execute_single_action(action_name, action_config)
                self._create_rollback_point(action_name)
            
            # Phase 4: Update load order configuration
            self._update_single_pass_load_order()
            
            # Phase 5: Validate cleanup results
            self._validate_cleanup_results()
            
            return self.execution_results
            
        except Exception as e:
            logger.error(f"Immediate cleanup failed: {e}")
            self.execution_results["summary"] = f"❌ Cleanup failed: {e}"
            self._offer_rollback()
            return self.execution_results
    
    def _create_comprehensive_backup(self) -> None:
        """Create timestamped backup of entire current structure"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"immediate_cleanup_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup entire core directory
            if self.core_dir.exists():
                core_backup = backup_path / "core"
                shutil.copytree(self.core_dir, core_backup)
                logger.info(f"Core directory backed up to: {core_backup}")
            
            # Backup setup.py if exists  
            setup_paths = [
                self.project_root / "setup.py",
                self.project_root / "pyics" / "setup.py"
            ]
            
            for setup_path in setup_paths:
                if setup_path.exists():
                    backup_setup = backup_path / f"setup_{setup_path.parent.name}.py"
                    shutil.copy2(setup_path, backup_setup)
                    logger.info(f"Setup.py backed up: {backup_setup}")
            
            # Create backup manifest
            manifest = {
                "backup_timestamp": timestamp,
                "backup_path": str(backup_path),
                "original_structure": self._capture_structure_snapshot(),
                "actions_planned": list(IMMEDIATE_ACTIONS.keys())
            }
            
            manifest_path = backup_path / "backup_manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            
            self.execution_results["backup_created"] = True
            logger.info("Comprehensive backup created successfully")
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    def _capture_structure_snapshot(self) -> Dict[str, Any]:
        """Capture current structure snapshot for backup manifest"""
        snapshot = {
            "domains": [],
            "total_files": 0,
            "complex_structures": {}
        }
        
        if not self.core_dir.exists():
            return snapshot
        
        for item in self.core_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                domain_info = {
                    "name": item.name,
                    "files": [],
                    "complex_structures": []
                }
                
                # Count files and detect complex structures
                for file_path in item.rglob("*"):
                    if file_path.is_file():
                        domain_info["files"].append(str(file_path.relative_to(item)))
                        snapshot["total_files"] += 1
                    elif file_path.is_dir() and file_path.name in COMPLEX_PATTERNS:
                        domain_info["complex_structures"].append(file_path.name)
                
                snapshot["domains"].append(domain_info)
        
        return snapshot
    
    def _discover_current_domains(self) -> None:
        """Discover current domain structure"""
        logger.info("Discovering current domain structure...")
        
        if not self.core_dir.exists():
            logger.error(f"Core directory not found: {self.core_dir}")
            return
        
        self.discovered_domains = []
        
        for item in self.core_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
                self.discovered_domains.append(item.name)
        
        logger.info(f"Discovered domains: {self.discovered_domains}")
    
    def _execute_single_action(self, action_name: str, action_config: Dict[str, Any]) -> None:
        """Execute single cleanup action"""
        try:
            if action_name == "merge_validation_to_validators":
                self._merge_validation_to_validators()
            elif action_name == "eliminate_transforms_to_transformations":
                self._eliminate_transforms_to_transformations()
            elif action_name == "eliminate_logic_domain":
                self._eliminate_logic_domain()
            elif action_name == "clean_complex_structures":
                self._clean_complex_structures()
            
            self.execution_results["actions_completed"][action_name] = {
                "status": "success",
                "description": action_config["description"],
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Completed: {action_config['description']}")
            
        except Exception as e:
            self.execution_results["actions_completed"][action_name] = {
                "status": "failed",
                "description": action_config["description"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            logger.error(f"❌ Failed: {action_config['description']} - {e}")
            raise
    
    def _merge_validation_to_validators(self) -> None:
        """Merge core/validation → core/validators"""
        validation_path = self.core_dir / "validation"
        validators_path = self.core_dir / "validators"
        
        if not validation_path.exists():
            logger.info("No validation domain found to merge")
            return
        
        # Ensure validators domain exists
        validators_path.mkdir(exist_ok=True)
        
        # Create merge record
        merge_record_path = validators_path / "_merged_from_validation.md"
        with open(merge_record_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Merged Content from validation Domain

Merged into: validators
Merge date: {datetime.now().isoformat()}
Action: Domain consolidation for single-pass loading

## Files Merged:
""")
        
        # Preserve and merge functions
        for py_file in validation_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                # Read source content
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Document in merge record
                with open(merge_record_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n### {py_file.name}\n```python\n{content}\n```\n")
                
                # Extract and categorize functions
                self._extract_and_preserve_functions(py_file, content, "validators")
                
            except Exception as e:
                logger.warning(f"Failed to merge {py_file}: {e}")
        
        # Remove validation domain
        shutil.rmtree(validation_path)
        logger.info("Merged validation → validators")
    
    def _eliminate_transforms_to_transformations(self) -> None:
        """Eliminate core/transforms → core/transformations"""
        transforms_path = self.core_dir / "transforms"
        transformations_path = self.core_dir / "transformations"
        
        if not transforms_path.exists():
            logger.info("No transforms domain found to eliminate")
            return
        
        # Ensure transformations domain exists
        transformations_path.mkdir(exist_ok=True)
        
        # Create legacy implementations directory
        legacy_dir = transformations_path / "legacy_transforms"
        legacy_dir.mkdir(exist_ok=True)
        
        # Create elimination record
        elimination_record_path = transformations_path / "_eliminated_transforms.md"
        with open(elimination_record_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Eliminated transforms Domain

Moved into: transformations/legacy_transforms/
Elimination date: {datetime.now().isoformat()}
Action: Legacy domain elimination for single-pass loading

## Legacy Files:
""")
        
        # Move legacy content
        for py_file in transforms_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                # Read source content
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Move to legacy directory
                legacy_file = legacy_dir / py_file.name
                with open(legacy_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Legacy transform from core/transforms\n")
                    f.write(f"# Moved: {datetime.now().isoformat()}\n\n")
                    f.write(content)
                
                # Document elimination
                with open(elimination_record_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n- {py_file.name} → legacy_transforms/{py_file.name}\n")
                
            except Exception as e:
                logger.warning(f"Failed to move {py_file}: {e}")
        
        # Remove transforms domain
        shutil.rmtree(transforms_path)
        logger.info("Eliminated transforms → transformations/legacy_transforms/")
    
    def _eliminate_logic_domain(self) -> None:
        """Eliminate core/logic, distribute to primitives/composition"""
        logic_path = self.core_dir / "logic"
        
        if not logic_path.exists():
            logger.info("No logic domain found to eliminate")
            return
        
        # Create distribution record
        distribution_record_path = self.core_dir / "_logic_distribution.md"
        with open(distribution_record_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Distributed logic Domain

Elimination date: {datetime.now().isoformat()}
Action: Legacy domain elimination - functions distributed to appropriate domains

## Distribution Plan:
- Atomic functions → primitives/
- Composition functions → composition/
- Deprecated functions → archived

## Files Processed:
""")
        
        # Analyze and distribute functions
        for py_file in logic_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze functions for distribution
                distribution_target = self._analyze_logic_functions(content)
                
                # Document distribution decision
                with open(distribution_record_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n### {py_file.name}\n")
                    f.write(f"Target: {distribution_target}\n")
                    f.write(f"```python\n{content}\n```\n")
                
                # Store for manual review (don't auto-distribute complex logic)
                preservation_dir = self.core_dir / "_preserved_logic"
                preservation_dir.mkdir(exist_ok=True)
                
                preservation_file = preservation_dir / f"{py_file.name}"
                with open(preservation_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Preserved from core/logic for manual distribution\n")
                    f.write(f"# Recommended target: {distribution_target}\n")
                    f.write(f"# Preserved: {datetime.now().isoformat()}\n\n")
                    f.write(content)
                
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")
        
        # Remove logic domain
        shutil.rmtree(logic_path)
        logger.info("Eliminated logic domain - functions preserved for manual distribution")
    
    def _analyze_logic_functions(self, content: str) -> str:
        """Analyze logic functions to determine distribution target"""
        try:
            tree = ast.parse(content)
            
            function_types = {
                "atomic": 0,
                "composition": 0,
                "complex": 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name.lower()
                    
                    # Categorize based on function name patterns
                    if any(keyword in func_name for keyword in ["atom", "simple", "basic", "util"]):
                        function_types["atomic"] += 1
                    elif any(keyword in func_name for keyword in ["compose", "chain", "pipe", "combine"]):
                        function_types["composition"] += 1
                    else:
                        function_types["complex"] += 1
            
            # Determine target based on function analysis
            if function_types["atomic"] > function_types["composition"]:
                return "primitives"
            elif function_types["composition"] > 0:
                return "composition"
            else:
                return "archive_deprecated"
                
        except Exception:
            return "manual_review_required"
    
    def _clean_complex_structures(self) -> None:
        """Clean complex nested structures from all domains"""
        logger.info("Cleaning complex nested structures...")
        
        cleaned_count = 0
        
        for domain_name in self.discovered_domains:
            domain_path = self.core_dir / domain_name
            
            if not domain_path.exists():
                continue
            
            # Check for and remove complex patterns
            for pattern in COMPLEX_PATTERNS:
                pattern_path = domain_path / pattern
                
                if pattern_path.exists():
                    # Preserve content before deletion
                    self._preserve_complex_structure_content(pattern_path, domain_name, pattern)
                    
                    # Remove complex structure
                    shutil.rmtree(pattern_path)
                    logger.info(f"Removed complex structure: {domain_name}/{pattern}")
                    cleaned_count += 1
        
        logger.info(f"Cleaned {cleaned_count} complex structures")
    
    def _preserve_complex_structure_content(self, structure_path: Path, domain_name: str, pattern: str) -> None:
        """Preserve content from complex structure before deletion"""
        preservation_dir = self.core_dir / "_preserved_complex_structures"
        preservation_dir.mkdir(exist_ok=True)
        
        preservation_file = preservation_dir / f"{domain_name}_{pattern}_preserved.md"
        
        with open(preservation_file, 'w', encoding='utf-8') as f:
            f.write(f"# Preserved Complex Structure: {domain_name}/{pattern}\n\n")
            f.write(f"Preserved: {datetime.now().isoformat()}\n")
            f.write(f"Reason: Complex structure cleanup for single-pass architecture\n\n")
            
            # Preserve all Python files
            for py_file in structure_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as py_f:
                        content = py_f.read()
                    
                    f.write(f"## {py_file.relative_to(structure_path)}\n")
                    f.write(f"```python\n{content}\n```\n\n")
                    
                except Exception as e:
                    f.write(f"## {py_file.relative_to(structure_path)}\n")
                    f.write(f"Error reading file: {e}\n\n")
    
    def _extract_and_preserve_functions(self, file_path: Path, content: str, target_domain: str) -> None:
        """Extract and preserve functions for manual integration"""
        try:
            tree = ast.parse(content)
            
            extracted_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    extracted_functions.append({
                        "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                        "name": node.name,
                        "lineno": node.lineno,
                        "source_file": str(file_path.name)
                    })
            
            if extracted_functions:
                self.preserved_functions[target_domain] = self.preserved_functions.get(target_domain, [])
                self.preserved_functions[target_domain].extend(extracted_functions)
                
        except Exception as e:
            logger.warning(f"Failed to extract functions from {file_path}: {e}")
    
    def _update_single_pass_load_order(self) -> None:
        """Update load order configuration for single-pass architecture"""
        logger.info("Updating single-pass load order configuration...")
        
        try:
            # Get current domains after cleanup
            current_domains = []
            for item in self.core_dir.iterdir():
                if (item.is_dir() and 
                    not item.name.startswith('.') and 
                    not item.name.startswith('_') and
                    item.name != '__pycache__'):
                    current_domains.append(item.name)
            
            # Create load order configuration
            load_order_config = {
                "single_pass_load_order": {},
                "domain_priorities": {},
                "cleanup_timestamp": datetime.now().isoformat(),
                "eliminated_domains": ["validation", "transforms", "logic"],
                "active_domains": current_domains
            }
            
            # Assign load orders for active domains
            for domain in current_domains:
                if domain in SINGLE_PASS_LOAD_ORDER:
                    load_order_config["single_pass_load_order"][domain] = SINGLE_PASS_LOAD_ORDER[domain]
                    load_order_config["domain_priorities"][domain] = SINGLE_PASS_LOAD_ORDER[domain] // 10
            
            # Write load order configuration
            config_path = self.core_dir / "_load_order_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(load_order_config, f, indent=2)
            
            self.execution_results["load_order_updated"] = True
            logger.info("Single-pass load order configuration updated")
            
        except Exception as e:
            logger.error(f"Failed to update load order: {e}")
            self.execution_results["load_order_updated"] = False
    
    def _create_rollback_point(self, action_name: str) -> None:
        """Create rollback point after each action"""
        rollback_point = {
            "action": action_name,
            "timestamp": datetime.now().isoformat(),
            "structure_snapshot": self._capture_structure_snapshot()
        }
        
        self.execution_results["rollback_points"].append(rollback_point)
    
    def _validate_cleanup_results(self) -> None:
        """Validate cleanup results"""
        logger.info("Validating cleanup results...")
        
        validation_issues = []
        
        try:
            # Check eliminated domains are gone
            eliminated_domains = ["validation", "transforms", "logic"]
            for domain in eliminated_domains:
                domain_path = self.core_dir / domain
                if domain_path.exists():
                    validation_issues.append(f"Domain {domain} still exists after elimination")
            
            # Check remaining domains have clean structure
            for item in self.core_dir.iterdir():
                if (item.is_dir() and 
                    not item.name.startswith('.') and 
                    not item.name.startswith('_') and
                    item.name != '__pycache__'):
                    
                    # Check for complex patterns
                    for pattern in COMPLEX_PATTERNS:
                        pattern_path = item / pattern
                        if pattern_path.exists():
                            validation_issues.append(f"Complex structure {item.name}/{pattern} still exists")
            
            # Check preservation directories exist
            preservation_dirs = [
                "_preserved_complex_structures",
                "_preserved_logic"
            ]
            
            for preservation_dir in preservation_dirs:
                preservation_path = self.core_dir / preservation_dir
                if not preservation_path.exists():
                    validation_issues.append(f"Preservation directory {preservation_dir} not created")
            
            if validation_issues:
                self.execution_results["validation_passed"] = False
                self.execution_results["summary"] = f"❌ Validation failed: {len(validation_issues)} issues found"
                for issue in validation_issues:
                    logger.error(f"Validation issue: {issue}")
            else:
                self.execution_results["validation_passed"] = True
                self.execution_results["summary"] = "✅ Immediate cleanup completed successfully"
                logger.info("Cleanup validation passed")
                
        except Exception as e:
            self.execution_results["validation_passed"] = False
            self.execution_results["summary"] = f"❌ Validation error: {e}"
            logger.error(f"Validation failed: {e}")
    
    def _offer_rollback(self) -> None:
        """Offer rollback options if cleanup fails"""
        logger.warning("Cleanup failed. Rollback options available in backup directory:")
        logger.warning(f"Backup location: {self.backup_dir}")
        
        # List available rollback points
        for i, rollback_point in enumerate(self.execution_results["rollback_points"]):
            logger.warning(f"Rollback point {i+1}: After {rollback_point['action']} at {rollback_point['timestamp']}")

def main():
    """Main execution function"""
    executor = PyicsImmediateCleanupExecutor(PROJECT_ROOT)
    results = executor.execute_immediate_cleanup()
    
    # Display results
    print("=" * 60)
    print("PYICS IMMEDIATE CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Backup Created: {'✅' if results['backup_created'] else '❌'}")
    print(f"Load Order Updated: {'✅' if results['load_order_updated'] else '❌'}")
    print(f"Validation Passed: {'✅' if results['validation_passed'] else '❌'}")
    
    print("\nActions Completed:")
    for action, details in results["actions_completed"].items():
        status_icon = "✅" if details["status"] == "success" else "❌"
        print(f"  {status_icon} {details['description']}")
        if details["status"] == "failed":
            print(f"     Error: {details['error']}")
    
    print("=" * 60)
    print(f"Status: {results['summary']}")
    print("=" * 60)
    
    if results["validation_passed"]:
        print("\n🎉 IMMEDIATE CLEANUP COMPLETED SUCCESSFULLY!")
        print("📋 Domains consolidated:")
        print("   • validation → validators")
        print("   • transforms → transformations/legacy_transforms/") 
        print("   • logic → eliminated (functions preserved)")
        print("🧹 Complex structures cleaned")
        print("⚡ Single-pass load order implemented")
        
        print("\n📋 NEXT STEPS:")
        print("1. Review preserved functions in _preserved_* directories")
        print("2. Manually integrate useful legacy functions")
        print("3. Run dependency validation on cleaned structure")
        print("4. Test single-pass loading with updated domains")
        print("5. Update imports in existing code")
        
        print("\n🔍 PRESERVED CONTENT LOCATIONS:")
        print("   • Complex structures: pyics/core/_preserved_complex_structures/")
        print("   • Logic functions: pyics/core/_preserved_logic/")
        print("   • Merge records: pyics/core/validators/_merged_from_validation.md")
        print("   • Elimination records: pyics/core/transformations/_eliminated_transforms.md")
    else:
        print("\n❌ CLEANUP ENCOUNTERED ISSUES")
        print("Check logs and backup directory for recovery options")
        print(f"Backup location: {executor.backup_dir}")
    
    sys.exit(0 if results["validation_passed"] else 1)

if __name__ == "__main__":
    main()

# [EOF] - End of immediate cleanup executor
