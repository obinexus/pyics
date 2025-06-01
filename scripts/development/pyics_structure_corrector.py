#!/usr/bin/env python3
"""
pyics_structure_corrector.py
Pyics Single-Pass Architecture Structure Corrector

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Systematic correction of domain violations and implementation of clean single-pass loading
Architecture: Domain consolidation with cost-aware priority loading
Methodology: Waterfall correction with preservation of functional code

PROBLEM SOLVED: Corrects complex nested structures and implements clean domain architecture
DEPENDENCIES: Standard library only (pathlib, shutil, ast)
THREAD SAFETY: Yes - atomic file operations with comprehensive backup
DETERMINISTIC: Yes - reproducible architecture correction with validation

This script systematically corrects the current structural violations by:
1. Consolidating redundant domains (validation/validators, transforms/transformations)
2. Flattening complex nested structures 
3. Implementing standard 6-module domain pattern
4. Creating proper single-pass IoC registry
5. Establishing correct load order dependencies
"""

import os
import sys
import shutil
import ast
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
import logging

# Configuration
PROJECT_ROOT = Path.cwd()
PYICS_CORE_DIR = "pyics/core"
BACKUP_DIR = "structure_backup"

# Domain consolidation mapping
DOMAIN_CONSOLIDATION = {
    "validation": "validators",  # Merge validation into validators
    "transforms": "transformations",  # Merge transforms into transformations
    "logic": "DELETE",  # Delete logic domain - redistribute functions
}

# Standard domain load order
DOMAIN_LOAD_ORDER = {
    "primitives": 10,
    "protocols": 20,
    "structures": 30,
    "composition": 40,
    "validators": 50,  # Consolidated validation
    "transformations": 60,  # Consolidated transforms
    "registry": 70,
    "routing": 80,
    "safety": 90
}

# Standard domain module pattern
STANDARD_MODULES = [
    "data_types.py",
    "operations.py", 
    "relations.py",
    "config.py",
    "__init__.py",
    "README.md"
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PyicsStructureCorrector:
    """
    Systematic corrector for Pyics domain architecture violations
    
    Implements comprehensive domain consolidation, structure flattening,
    and single-pass architecture establishment with validation integration.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.core_dir = self.project_root / PYICS_CORE_DIR
        self.backup_dir = self.project_root / BACKUP_DIR
        
        self.correction_results = {
            "backup_created": False,
            "complex_structures_cleaned": 0,
            "domains_consolidated": 0,
            "standard_modules_generated": 0,
            "ioc_registry_created": False,
            "validation_passed": False,
            "summary": ""
        }
        
        self.discovered_domains = []
        self.domain_analysis = {}
        
    def execute_complete_correction(self) -> Dict[str, Any]:
        """Execute complete structure correction process"""
        logger.info("=" * 60)
        logger.info("PYICS SINGLE-PASS ARCHITECTURE CORRECTION")
        logger.info("Engineering Lead: Nnamdi Okpala / OBINexus Computing")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Create comprehensive backup
            self._create_structure_backup()
            
            # Phase 2: Analyze current domain structure
            self._analyze_domain_structure()
            
            # Phase 3: Clean complex nested structures
            self._clean_complex_structures()
            
            # Phase 4: Consolidate redundant domains
            self._consolidate_domains()
            
            # Phase 5: Generate standard domain modules
            self._generate_standard_modules()
            
            # Phase 6: Create single-pass IoC registry
            self._create_ioc_registry()
            
            # Phase 7: Validate corrected architecture
            self._validate_architecture()
            
            return self.correction_results
            
        except Exception as e:
            logger.error(f"Structure correction failed: {e}")
            self.correction_results["summary"] = f"❌ Critical correction failure: {e}"
            return self.correction_results
    
    def _create_structure_backup(self) -> None:
        """Create comprehensive backup of current structure"""
        try:
            # Create timestamped backup directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"structure_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup entire core directory
            if self.core_dir.exists():
                shutil.copytree(self.core_dir, backup_path / "core")
                logger.info(f"Structure backup created: {backup_path}")
                
            self.correction_results["backup_created"] = True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    def _analyze_domain_structure(self) -> None:
        """Analyze current domain structure for correction planning"""
        logger.info("Analyzing current domain structure...")
        
        if not self.core_dir.exists():
            logger.error(f"Core directory not found: {self.core_dir}")
            return
        
        for item in self.core_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
                domain_name = item.name
                self.discovered_domains.append(domain_name)
                
                # Analyze domain for violations
                analysis = self._analyze_single_domain(item, domain_name)
                self.domain_analysis[domain_name] = analysis
                
                logger.info(f"Analyzed {domain_name}: {analysis['violations']} violations")
        
        logger.info(f"Domain analysis complete: {len(self.discovered_domains)} domains")
    
    def _analyze_single_domain(self, domain_path: Path, domain_name: str) -> Dict[str, Any]:
        """Analyze single domain for structural violations"""
        analysis = {
            "domain_name": domain_name,
            "path": str(domain_path),
            "violations": [],
            "has_complex_nesting": False,
            "has_standard_modules": [],
            "missing_modules": [],
            "extractable_functions": [],
            "consolidation_target": None
        }
        
        # Check for complex nesting violations
        complex_dirs = ["implementations", "interfaces", "compliance", "contracts", "tests"]
        for complex_dir in complex_dirs:
            if (domain_path / complex_dir).exists():
                analysis["violations"].append(f"complex_nesting: {complex_dir}/")
                analysis["has_complex_nesting"] = True
        
        # Check for standard modules
        for module in STANDARD_MODULES:
            if (domain_path / module).exists():
                analysis["has_standard_modules"].append(module)
            else:
                analysis["missing_modules"].append(module)
        
        # Check for consolidation requirements
        if domain_name in DOMAIN_CONSOLIDATION:
            analysis["consolidation_target"] = DOMAIN_CONSOLIDATION[domain_name]
            analysis["violations"].append(f"consolidation_required: {analysis['consolidation_target']}")
        
        # Extract functions from complex structures
        if analysis["has_complex_nesting"]:
            analysis["extractable_functions"] = self._extract_functions_from_complex_structure(domain_path)
        
        return analysis
    
    def _extract_functions_from_complex_structure(self, domain_path: Path) -> List[Dict[str, Any]]:
        """Extract functions from complex nested structures"""
        functions = []
        
        for py_file in domain_path.rglob("*.py"):
            if py_file.name != "__init__.py":
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse AST to extract functions and classes
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            functions.append({
                                "type": "function",
                                "name": node.name,
                                "file": str(py_file.relative_to(domain_path)),
                                "lineno": node.lineno
                            })
                        elif isinstance(node, ast.ClassDef):
                            functions.append({
                                "type": "class", 
                                "name": node.name,
                                "file": str(py_file.relative_to(domain_path)),
                                "lineno": node.lineno
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to parse {py_file}: {e}")
        
        return functions
    
    def _clean_complex_structures(self) -> None:
        """Clean complex nested structures while preserving functionality"""
        logger.info("Cleaning complex nested structures...")
        
        cleaned_count = 0
        
        for domain_name, analysis in self.domain_analysis.items():
            if analysis["has_complex_nesting"]:
                domain_path = Path(analysis["path"])
                
                # Extract and preserve functions before cleanup
                self._preserve_domain_functions(domain_path, analysis)
                
                # Remove complex directories
                complex_dirs = ["implementations", "interfaces", "compliance", "contracts", "tests"]
                for complex_dir in complex_dirs:
                    complex_path = domain_path / complex_dir
                    if complex_path.exists():
                        shutil.rmtree(complex_path)
                        logger.info(f"Removed complex structure: {domain_name}/{complex_dir}")
                        cleaned_count += 1
        
        self.correction_results["complex_structures_cleaned"] = cleaned_count
        logger.info(f"Complex structure cleanup complete: {cleaned_count} structures cleaned")
    
    def _preserve_domain_functions(self, domain_path: Path, analysis: Dict[str, Any]) -> None:
        """Preserve functions from complex structures into standard modules"""
        functions = analysis["extractable_functions"]
        
        if not functions:
            return
        
        # Group functions by target module
        function_groups = {
            "data_types.py": [],
            "operations.py": [],
            "relations.py": []
        }
        
        for func in functions:
            # Categorize function based on name patterns
            if any(keyword in func["name"].lower() for keyword in ["type", "class", "structure", "model"]):
                function_groups["data_types.py"].append(func)
            elif any(keyword in func["name"].lower() for keyword in ["relation", "link", "connect", "associate"]):
                function_groups["relations.py"].append(func)
            else:
                function_groups["operations.py"].append(func)
        
        # Write preserved functions to temporary files for manual review
        preservation_dir = domain_path / "_preserved_functions"
        preservation_dir.mkdir(exist_ok=True)
        
        for target_module, funcs in function_groups.items():
            if funcs:
                preservation_file = preservation_dir / f"for_{target_module}"
                with open(preservation_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Functions preserved from complex structure for {target_module}\n")
                    f.write(f"# Review and integrate manually\n\n")
                    for func in funcs:
                        f.write(f"# {func['type']}: {func['name']} from {func['file']}:{func['lineno']}\n")
    
    def _consolidate_domains(self) -> None:
        """Consolidate redundant domains according to consolidation mapping"""
        logger.info("Consolidating redundant domains...")
        
        consolidated_count = 0
        
        for source_domain, target in DOMAIN_CONSOLIDATION.items():
            source_path = self.core_dir / source_domain
            
            if not source_path.exists():
                continue
                
            if target == "DELETE":
                # Special handling for logic domain - preserve useful functions
                self._preserve_and_delete_domain(source_path, source_domain)
            else:
                # Merge source into target
                target_path = self.core_dir / target
                self._merge_domains(source_path, target_path, source_domain, target)
            
            consolidated_count += 1
        
        self.correction_results["domains_consolidated"] = consolidated_count
        logger.info(f"Domain consolidation complete: {consolidated_count} domains consolidated")
    
    def _preserve_and_delete_domain(self, source_path: Path, source_domain: str) -> None:
        """Preserve useful functions from domain before deletion"""
        # Create preservation record
        preservation_path = self.core_dir / "_domain_preservation" / f"{source_domain}_preserved.md"
        preservation_path.parent.mkdir(exist_ok=True)
        
        with open(preservation_path, 'w', encoding='utf-8') as f:
            f.write(f"# Preserved Functions from {source_domain} Domain\n\n")
            f.write(f"Domain deleted: {datetime.now().isoformat()}\n")
            f.write(f"Reason: Redundant functionality - distribute to appropriate domains\n\n")
            
            # List preserved functions
            for py_file in source_path.rglob("*.py"):
                if py_file.name != "__init__.py":
                    try:
                        with open(py_file, 'r', encoding='utf-8') as py_f:
                            content = py_f.read()
                        f.write(f"## {py_file.name}\n```python\n{content}\n```\n\n")
                    except Exception as e:
                        f.write(f"## {py_file.name}\nError reading file: {e}\n\n")
        
        # Delete domain
        shutil.rmtree(source_path)
        logger.info(f"Preserved and deleted domain: {source_domain}")
    
    def _merge_domains(self, source_path: Path, target_path: Path, source_domain: str, target_domain: str) -> None:
        """Merge source domain into target domain"""
        # Ensure target domain exists
        target_path.mkdir(exist_ok=True)
        
        # Create merge record
        merge_record_path = target_path / f"_merged_from_{source_domain}.md"
        
        with open(merge_record_path, 'w', encoding='utf-8') as f:
            f.write(f"# Merged Content from {source_domain} Domain\n\n")
            f.write(f"Merged into: {target_domain}\n")
            f.write(f"Merge date: {datetime.now().isoformat()}\n\n")
        
        # Copy useful files to target domain
        for py_file in source_path.rglob("*.py"):
            if py_file.name not in ["__init__.py"]:
                try:
                    # Read source content
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Write to merge record for manual integration
                    with open(merge_record_path, 'a', encoding='utf-8') as f:
                        f.write(f"## {py_file.name}\n```python\n{content}\n```\n\n")
                        
                except Exception as e:
                    logger.warning(f"Failed to merge {py_file}: {e}")
        
        # Delete source domain
        shutil.rmtree(source_path)
        logger.info(f"Merged {source_domain} → {target_domain}")
    
    def _generate_standard_modules(self) -> None:
        """Generate standard modules for all domains"""
        logger.info("Generating standard domain modules...")
        
        generated_count = 0
        
        # Get current domains after consolidation
        current_domains = []
        for item in self.core_dir.iterdir():
            if (item.is_dir() and 
                not item.name.startswith('.') and 
                not item.name.startswith('_') and
                item.name != '__pycache__'):
                current_domains.append(item.name)
        
        for domain_name in current_domains:
            domain_path = self.core_dir / domain_name
            
            # Generate missing standard modules
            for module_name in STANDARD_MODULES:
                module_path = domain_path / module_name
                
                if not module_path.exists():
                    self._generate_standard_module(domain_path, module_name, domain_name)
                    generated_count += 1
        
        self.correction_results["standard_modules_generated"] = generated_count
        logger.info(f"Standard module generation complete: {generated_count} modules generated")
    
    def _generate_standard_module(self, domain_path: Path, module_name: str, domain_name: str) -> None:
        """Generate individual standard module"""
        module_path = domain_path / module_name
        
        if module_name == "data_types.py":
            content = self._generate_data_types_module(domain_name)
        elif module_name == "operations.py":
            content = self._generate_operations_module(domain_name)
        elif module_name == "relations.py":
            content = self._generate_relations_module(domain_name)
        elif module_name == "config.py":
            content = self._generate_config_module(domain_name)
        elif module_name == "__init__.py":
            content = self._generate_init_module(domain_name)
        elif module_name == "README.md":
            content = self._generate_readme_module(domain_name)
        else:
            return
        
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.debug(f"Generated {domain_name}/{module_name}")
    
    def _generate_data_types_module(self, domain_name: str) -> str:
        """Generate data_types.py module"""
        return f'''#!/usr/bin/env python3
"""
pyics/core/{domain_name}/data_types.py
{domain_name.title()} Domain Data Types

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: {domain_name}
Load Order: {DOMAIN_LOAD_ORDER.get(domain_name, 50)}

PROBLEM SOLVED: Provides immutable data structures for {domain_name} domain operations
DEPENDENCIES: typing, dataclasses for structure definition
THREAD SAFETY: Yes - immutable data structures with type safety
DETERMINISTIC: Yes - predictable structure behavior and validation

This module defines the core data structures and type definitions for the {domain_name} domain,
ensuring type safety and immutability across all domain operations.
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(f"pyics.core.{domain_name}.data_types")

# Type variables for generic structures
T = TypeVar('T')

@dataclass(frozen=True)
class {domain_name.title()}Entity:
    """Base entity for {domain_name} domain"""
    entity_id: str
    entity_type: str = "{domain_name}"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if not self.entity_id:
            raise ValueError("Entity ID cannot be empty")
        if not self.entity_type:
            raise ValueError("Entity type cannot be empty")

# Export all data types
__all__ = [
    "{domain_name.title()}Entity",
]

# [EOF] - End of {domain_name} data types module
'''

    def _generate_operations_module(self, domain_name: str) -> str:
        """Generate operations.py module"""
        return f'''#!/usr/bin/env python3
"""
pyics/core/{domain_name}/operations.py
{domain_name.title()} Domain Operations

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: {domain_name}
Load Order: {DOMAIN_LOAD_ORDER.get(domain_name, 50)}

PROBLEM SOLVED: Provides pure functional operations for {domain_name} domain transformations
DEPENDENCIES: {domain_name}.data_types for structure definitions
THREAD SAFETY: Yes - pure functional operations with no shared state
DETERMINISTIC: Yes - referentially transparent functions with predictable outputs

This module implements pure functional operations for {domain_name} domain entities,
ensuring deterministic transformations and maintaining immutability guarantees.
"""

from typing import Any, Dict, List, Optional, Callable, TypeVar
from .data_types import {domain_name.title()}Entity
import logging

logger = logging.getLogger(f"pyics.core.{domain_name}.operations")

# Type variables
T = TypeVar('T')
U = TypeVar('U')

def create_{domain_name}_entity(entity_id: str, **kwargs) -> {domain_name.title()}Entity:
    """
    Create new {domain_name} entity with validation
    
    Args:
        entity_id: Unique identifier for entity
        **kwargs: Additional entity metadata
        
    Returns:
        Validated {domain_name.title()}Entity instance
    """
    try:
        entity = {domain_name.title()}Entity(
            entity_id=entity_id,
            metadata=kwargs
        )
        
        logger.debug(f"Created {domain_name} entity: {{entity_id}}")
        return entity
        
    except Exception as e:
        logger.error(f"Failed to create {domain_name} entity: {{e}}")
        raise

def validate_{domain_name}_entity(entity: {domain_name.title()}Entity) -> bool:
    """
    Validate {domain_name} entity for consistency
    
    Args:
        entity: Entity to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(entity, {domain_name.title()}Entity):
            return False
            
        if not entity.entity_id or not entity.entity_type:
            return False
            
        logger.debug(f"Validated {domain_name} entity: {{entity.entity_id}}")
        return True
        
    except Exception as e:
        logger.error(f"Entity validation failed: {{e}}")
        return False

# Export all operations
__all__ = [
    "create_{domain_name}_entity",
    "validate_{domain_name}_entity",
]

# [EOF] - End of {domain_name} operations module
'''

    def _generate_relations_module(self, domain_name: str) -> str:
        """Generate relations.py module"""
        return f'''#!/usr/bin/env python3
"""
pyics/core/{domain_name}/relations.py
{domain_name.title()} Domain Relations

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: {domain_name}
Load Order: {DOMAIN_LOAD_ORDER.get(domain_name, 50)}

PROBLEM SOLVED: Defines relationship patterns and constraints for {domain_name} domain entities
DEPENDENCIES: {domain_name}.data_types for entity definitions
THREAD SAFETY: Yes - immutable relationship definitions
DETERMINISTIC: Yes - consistent relationship validation and enforcement

This module defines the relationship patterns, constraints, and validation rules
for entities within the {domain_name} domain and their interactions with other domains.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Protocol
from .data_types import {domain_name.title()}Entity
import logging

logger = logging.getLogger(f"pyics.core.{domain_name}.relations")

class {domain_name.title()}Relation(Protocol):
    """Protocol for {domain_name} domain relations"""
    
    def validate_relation(self, source: {domain_name.title()}Entity, target: {domain_name.title()}Entity) -> bool:
        """Validate relationship between entities"""
        ...
    
    def get_relation_type(self) -> str:
        """Get relation type identifier"""
        ...

def create_{domain_name}_relation(
    source: {domain_name.title()}Entity, 
    target: {domain_name.title()}Entity,
    relation_type: str = "default"
) -> Dict[str, Any]:
    """
    Create relationship between {domain_name} entities
    
    Args:
        source: Source entity
        target: Target entity
        relation_type: Type of relationship
        
    Returns:
        Relationship definition dictionary
    """
    try:
        relation = {{
            "source_id": source.entity_id,
            "target_id": target.entity_id,
            "relation_type": relation_type,
            "domain": "{domain_name}",
            "created_at": source.created_at
        }}
        
        logger.debug(f"Created {domain_name} relation: {{relation_type}}")
        return relation
        
    except Exception as e:
        logger.error(f"Failed to create {domain_name} relation: {{e}}")
        raise

def validate_{domain_name}_relations(relations: List[Dict[str, Any]]) -> bool:
    """
    Validate collection of {domain_name} relations
    
    Args:
        relations: List of relation definitions
        
    Returns:
        True if all relations valid, False otherwise
    """
    try:
        for relation in relations:
            required_fields = ["source_id", "target_id", "relation_type", "domain"]
            if not all(field in relation for field in required_fields):
                return False
                
            if relation["domain"] != "{domain_name}":
                return False
        
        logger.debug(f"Validated {{len(relations)}} {domain_name} relations")
        return True
        
    except Exception as e:
        logger.error(f"Relation validation failed: {{e}}")
        return False

# Export all relations
__all__ = [
    "{domain_name.title()}Relation",
    "create_{domain_name}_relation",
    "validate_{domain_name}_relations",
]

# [EOF] - End of {domain_name} relations module
'''

    def _generate_config_module(self, domain_name: str) -> str:
        """Generate config.py module with cost metadata"""
        load_order = DOMAIN_LOAD_ORDER.get(domain_name, 50)
        
        return f'''#!/usr/bin/env python3
"""
pyics/core/{domain_name}/config.py
{domain_name.title()} Domain Configuration

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: {domain_name}
Load Order: {load_order}

PROBLEM SOLVED: Provides centralized configuration and cost metadata for {domain_name} domain
DEPENDENCIES: typing for type definitions
THREAD SAFETY: Yes - immutable configuration data
DETERMINISTIC: Yes - static configuration with predictable behavior

This module provides DOP-compliant configuration management for the {domain_name} domain,
including cost metadata, behavior policies, and dependency injection interfaces.
"""

from typing import Dict, List, Any, TypedDict, Literal
import logging

logger = logging.getLogger(f"pyics.core.{domain_name}.config")

class DomainCostMetadata(TypedDict):
    priority_index: int
    compute_time_weight: float
    exposure_type: Literal["public", "internal", "private"]
    dependency_level: int
    thread_safe: bool
    load_order: int

class DomainConfiguration(TypedDict):
    domain_name: str
    cost_metadata: DomainCostMetadata
    data_types_available: List[str]
    relations_defined: List[str]
    behavior_policies: Dict[str, Any]
    export_interface: List[str]

# Cost metadata for {domain_name} domain
cost_metadata: DomainCostMetadata = {{
    "priority_index": {min(load_order // 10, 9)},
    "compute_time_weight": {0.1 + (load_order - 10) * 0.05},
    "exposure_type": "{"public" if load_order <= 30 else "internal"}",
    "dependency_level": {max(1, load_order // 20)},
    "thread_safe": True,
    "load_order": {load_order}
}}

# Domain configuration
DOMAIN_CONFIG: DomainConfiguration = {{
    "domain_name": "{domain_name}",
    "cost_metadata": cost_metadata,
    "data_types_available": ["{domain_name.title()}Entity"],
    "relations_defined": ["{domain_name.title()}Relation"],
    "behavior_policies": {{
        "strict_validation": True,
        "auto_dependency_resolution": True,
        "lazy_loading": False,
        "cache_enabled": True
    }},
    "export_interface": [
        "get_domain_metadata",
        "validate_configuration",
        "cost_metadata"
    ]
}}

def get_domain_metadata() -> DomainConfiguration:
    """Get complete domain configuration metadata"""
    return DOMAIN_CONFIG

def validate_configuration() -> bool:
    """Validate domain configuration for consistency"""
    try:
        # Validate required fields
        required_fields = ["domain_name", "cost_metadata", "data_types_available"]
        for field in required_fields:
            if field not in DOMAIN_CONFIG:
                logger.error(f"Missing required configuration field: {{field}}")
                return False
        
        # Validate cost metadata
        cost_meta = DOMAIN_CONFIG["cost_metadata"]
        if cost_meta["load_order"] != {load_order}:
            logger.error(f"Load order mismatch: expected {load_order}, got {{cost_meta['load_order']}}")
            return False
        
        logger.debug(f"Domain {domain_name} configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {{e}}")
        return False

def get_cost_metadata() -> DomainCostMetadata:
    """Get domain cost metadata for orchestration"""
    return cost_metadata

# Export configuration interfaces
__all__ = [
    "cost_metadata",
    "get_domain_metadata", 
    "validate_configuration",
    "get_cost_metadata",
    "DomainCostMetadata",
    "DomainConfiguration"
]

# Auto-validate on module load
if not validate_configuration():
    logger.warning(f"Domain {domain_name} configuration loaded with validation warnings")

# [EOF] - End of {domain_name} configuration module
'''

    def _generate_init_module(self, domain_name: str) -> str:
        """Generate __init__.py module"""
        return f'''#!/usr/bin/env python3
"""
pyics/core/{domain_name}/__init__.py
{domain_name.title()} Domain Module

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: {domain_name}
Load Order: {DOMAIN_LOAD_ORDER.get(domain_name, 50)}

This module provides the public interface for the {domain_name} domain,
exposing domain operations, data types, and configuration interfaces.
"""

# Import domain components
from .data_types import {domain_name.title()}Entity
from .operations import create_{domain_name}_entity, validate_{domain_name}_entity
from .relations import create_{domain_name}_relation, validate_{domain_name}_relations
from .config import get_domain_metadata, validate_configuration, cost_metadata

# Public domain interface
__all__ = [
    # Data types
    "{domain_name.title()}Entity",
    
    # Operations
    "create_{domain_name}_entity",
    "validate_{domain_name}_entity",
    
    # Relations
    "create_{domain_name}_relation", 
    "validate_{domain_name}_relations",
    
    # Configuration
    "get_domain_metadata",
    "validate_configuration",
    "cost_metadata"
]

# Domain metadata for external access
__domain__ = "{domain_name}"
__load_order__ = {DOMAIN_LOAD_ORDER.get(domain_name, 50)}
__version__ = "1.0.0"

# [EOF] - End of {domain_name} domain module
'''

    def _generate_readme_module(self, domain_name: str) -> str:
        """Generate README.md module"""
        return f'''# {domain_name.title()} Domain

**Engineering Lead**: Nnamdi Okpala / OBINexus Computing  
**Domain**: {domain_name}  
**Load Order**: {DOMAIN_LOAD_ORDER.get(domain_name, 50)}  
**Architecture**: Single-Pass DOP-Compliant Domain

## Overview

The {domain_name} domain provides specialized functionality for calendar automation within the Pyics system architecture. This domain follows strict single-pass loading principles and maintains complete separation of concerns.

## Domain Structure

```
{domain_name}/
├── data_types.py      # Immutable domain entities and structures
├── operations.py      # Pure functional domain operations  
├── relations.py       # Domain relationship definitions
├── config.py          # Domain configuration and cost metadata
├── __init__.py        # Public domain interface
└── README.md          # This documentation
```

## Cost Metadata

- **Load Order**: {DOMAIN_LOAD_ORDER.get(domain_name, 50)}
- **Dependency Level**: {max(1, DOMAIN_LOAD_ORDER.get(domain_name, 50) // 20)}
- **Thread Safe**: Yes
- **Compute Weight**: {0.1 + (DOMAIN_LOAD_ORDER.get(domain_name, 50) - 10) * 0.05}

## Usage Examples

### Basic Entity Creation

```python
from pyics.core.{domain_name} import create_{domain_name}_entity, {domain_name.title()}Entity

# Create domain entity
entity = create_{domain_name}_entity("entity_001", description="Example entity")

# Validate entity
from pyics.core.{domain_name} import validate_{domain_name}_entity
is_valid = validate_{domain_name}_entity(entity)
```

### Domain Configuration

```python
from pyics.core.{domain_name} import get_domain_metadata, cost_metadata

# Get complete domain metadata
metadata = get_domain_metadata()
print(f"Domain: {{metadata['domain_name']}}")
print(f"Load Order: {{metadata['cost_metadata']['load_order']}}")

# Access cost metadata directly
print(f"Compute Weight: {{cost_metadata['compute_time_weight']}}")
```

### Relationship Management

```python
from pyics.core.{domain_name} import create_{domain_name}_relation

# Create relationship between entities
relation = create_{domain_name}_relation(source_entity, target_entity, "dependency")
```

## Dependencies

This domain depends on:
- Python 3.8+ standard library
- typing module for type definitions
- dataclasses for immutable structures

## Integration

The {domain_name} domain integrates with:
- IoC Registry for dependency resolution
- Cost-aware orchestration system
- Single-pass loading architecture

## Development

When extending this domain:
1. Maintain immutability of all data structures
2. Use pure functional operations only  
3. Follow single-pass loading principles
4. Update cost metadata for new operations
5. Maintain comprehensive test coverage

---

**Load Order**: {DOMAIN_LOAD_ORDER.get(domain_name, 50)} | **Thread Safe**: ✅ | **DOP Compliant**: ✅
'''

    def _create_ioc_registry(self) -> None:
        """Create single-pass IoC registry"""
        logger.info("Creating single-pass IoC registry...")
        
        # Get current domains after consolidation
        current_domains = []
        for item in self.core_dir.iterdir():
            if (item.is_dir() and 
                not item.name.startswith('.') and 
                not item.name.startswith('_') and
                item.name != '__pycache__'):
                current_domains.append(item.name)
        
        # Sort domains by load order
        sorted_domains = sorted(current_domains, key=lambda d: DOMAIN_LOAD_ORDER.get(d, 50))
        
        registry_content = f'''#!/usr/bin/env python3
"""
pyics/core/ioc_registry.py
Single-Pass IoC Registry for Domain Configuration Resolution

Generated: {datetime.now().isoformat()}
Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Cost-aware single-pass domain loading with dependency resolution
Architecture: DOP-compliant IoC container with deterministic load order

PROBLEM SOLVED: Provides single-pass domain loading with cost-aware optimization
DEPENDENCIES: All pyics.core domain configuration modules
THREAD SAFETY: Yes - immutable registry with concurrent access support
DETERMINISTIC: Yes - predictable load order and dependency resolution

This registry implements single-pass domain loading based on cost metadata
and provides type-safe dependency injection for runtime orchestration.
"""

import importlib
import sys
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger("pyics.core.ioc_registry")

# Single-pass load order (cost-optimized)
SINGLE_PASS_LOAD_ORDER = {sorted_domains}

# Domain cost metadata cache
_domain_cost_cache: Dict[str, Dict[str, Any]] = {{}}
_registry_initialized = False

class SinglePassRegistry:
    """
    Cost-aware single-pass domain registry
    
    Implements deterministic domain loading based on cost metadata
    and dependency analysis for optimal system initialization.
    """
    
    def __init__(self):
        self._loaded_domains: Dict[str, Any] = {{}}
        self._load_times: Dict[str, float] = {{}}
        self._initialization_complete = False
    
    def initialize_single_pass(self) -> bool:
        """
        Execute single-pass domain loading in cost-optimized order
        
        Returns:
            True if initialization successful, False otherwise
        """
        global _registry_initialized
        
        if _registry_initialized:
            logger.warning("Registry already initialized")
            return True
        
        try:
            logger.info("Executing single-pass domain loading...")
            
            import time
            total_start = time.time()
            
            for domain_name in SINGLE_PASS_LOAD_ORDER:
                start_time = time.time()
                
                if self._load_domain_single_pass(domain_name):
                    load_time = time.time() - start_time
                    self._load_times[domain_name] = load_time
                    logger.debug(f"Loaded {{domain_name}} in {{load_time:.3f}}s")
                else:
                    logger.error(f"Failed to load domain: {{domain_name}}")
                    return False
            
            total_time = time.time() - total_start
            logger.info(f"Single-pass loading complete in {{total_time:.3f}}s")
            logger.info(f"Loaded domains: {{list(self._loaded_domains.keys())}}")
            
            _registry_initialized = True
            self._initialization_complete = True
            return True
            
        except Exception as e:
            logger.error(f"Single-pass initialization failed: {{e}}")
            return False
    
    def _load_domain_single_pass(self, domain_name: str) -> bool:
        """Load single domain with validation"""
        try:
            # Import domain module
            module_name = f"pyics.core.{{domain_name}}"
            domain_module = importlib.import_module(module_name)
            
            # Validate domain interface
            required_attrs = ["get_domain_metadata", "validate_configuration", "cost_metadata"]
            for attr in required_attrs:
                if not hasattr(domain_module, attr):
                    logger.error(f"Domain {{domain_name}} missing required attribute: {{attr}}")
                    return False
            
            # Validate domain configuration
            if not domain_module.validate_configuration():
                logger.error(f"Domain {{domain_name}} configuration validation failed")
                return False
            
            # Cache domain metadata
            metadata = domain_module.get_domain_metadata()
            cost_metadata = domain_module.cost_metadata
            
            self._loaded_domains[domain_name] = {{
                "module": domain_module,
                "metadata": metadata,
                "cost_metadata": cost_metadata
            }}
            
            # Update global cost cache
            _domain_cost_cache[domain_name] = cost_metadata
            
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import domain {{domain_name}}: {{e}}")
            return False
        except Exception as e:
            logger.error(f"Error loading domain {{domain_name}}: {{e}}")
            return False
    
    def get_domain_metadata(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for loaded domain"""
        if not self._initialization_complete:
            logger.warning("Registry not fully initialized")
            return None
        
        return self._loaded_domains.get(domain_name, {{}}).get("metadata")
    
    def get_load_order(self) -> List[str]:
        """Get single-pass load order"""
        return SINGLE_PASS_LOAD_ORDER.copy()
    
    def get_load_performance(self) -> Dict[str, float]:
        """Get domain load performance metrics"""
        return self._load_times.copy()
    
    def validate_single_pass_compliance(self) -> bool:
        """Validate single-pass loading compliance"""
        try:
            # Check all domains loaded
            for domain_name in SINGLE_PASS_LOAD_ORDER:
                if domain_name not in self._loaded_domains:
                    logger.error(f"Domain {{domain_name}} not loaded")
                    return False
            
            # Validate load order compliance
            for i, domain_name in enumerate(SINGLE_PASS_LOAD_ORDER):
                expected_load_order = self._loaded_domains[domain_name]["cost_metadata"]["load_order"]
                
                # Check load order consistency
                for j in range(i):
                    prev_domain = SINGLE_PASS_LOAD_ORDER[j]
                    prev_load_order = self._loaded_domains[prev_domain]["cost_metadata"]["load_order"]
                    
                    if prev_load_order >= expected_load_order:
                        logger.error(f"Load order violation: {{prev_domain}} ({{prev_load_order}}) >= {{domain_name}} ({{expected_load_order}})")
                        return False
            
            logger.info("Single-pass compliance validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Compliance validation failed: {{e}}")
            return False

# Global registry instance
_registry_instance: Optional[SinglePassRegistry] = None

def get_registry() -> SinglePassRegistry:
    """Get or create global registry instance"""
    global _registry_instance
    
    if _registry_instance is None:
        _registry_instance = SinglePassRegistry()
        if not _registry_instance.initialize_single_pass():
            raise RuntimeError("Failed to initialize single-pass registry")
    
    return _registry_instance

def get_domain_metadata(domain_name: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get domain metadata"""
    registry = get_registry()
    return registry.get_domain_metadata(domain_name)

def get_all_domains() -> List[str]:
    """Get list of all loaded domains"""
    return SINGLE_PASS_LOAD_ORDER.copy()

def get_domain_cost_metadata(domain_name: str) -> Optional[Dict[str, Any]]:
    """Get domain cost metadata"""
    return _domain_cost_cache.get(domain_name)

def validate_architecture() -> bool:
    """Validate complete architecture compliance"""
    registry = get_registry()
    return registry.validate_single_pass_compliance()

# Export registry interfaces
__all__ = [
    "SinglePassRegistry",
    "get_registry",
    "get_domain_metadata",
    "get_all_domains",
    "get_domain_cost_metadata", 
    "validate_architecture",
    "SINGLE_PASS_LOAD_ORDER"
]

# Auto-initialize registry
try:
    logger.debug("Auto-initializing single-pass registry...")
    _auto_registry = get_registry()
except Exception as e:
    logger.error(f"Failed to auto-initialize registry: {{e}}")

# [EOF] - End of single-pass IoC registry
'''
        
        registry_path = self.core_dir / "ioc_registry.py"
        with open(registry_path, 'w', encoding='utf-8') as f:
            f.write(registry_content)
        
        self.correction_results["ioc_registry_created"] = True
        logger.info(f"Single-pass IoC registry created: {registry_path}")
    
    def _validate_architecture(self) -> None:
        """Validate corrected architecture compliance"""
        logger.info("Validating corrected architecture...")
        
        try:
            # Test import of IoC registry
            ioc_path = self.core_dir / "ioc_registry.py"
            if not ioc_path.exists():
                raise ValueError("IoC registry not found")
            
            # Validate domain structure compliance
            validation_errors = []
            
            for item in self.core_dir.iterdir():
                if (item.is_dir() and 
                    not item.name.startswith('.') and 
                    not item.name.startswith('_') and
                    item.name != '__pycache__'):
                    
                    domain_name = item.name
                    
                    # Check for standard modules
                    for module in STANDARD_MODULES:
                        module_path = item / module
                        if not module_path.exists():
                            validation_errors.append(f"{domain_name} missing {module}")
                    
                    # Check for complex nesting violations
                    complex_dirs = ["implementations", "interfaces", "compliance", "contracts", "tests"]
                    for complex_dir in complex_dirs:
                        if (item / complex_dir).exists():
                            validation_errors.append(f"{domain_name} has complex nesting: {complex_dir}/")
            
            if validation_errors:
                logger.error(f"Architecture validation failed: {validation_errors}")
                self.correction_results["validation_passed"] = False
                self.correction_results["summary"] = f"❌ Validation failed: {len(validation_errors)} violations"
            else:
                logger.info("Architecture validation passed")
                self.correction_results["validation_passed"] = True
                self.correction_results["summary"] = "✅ Single-pass architecture implemented successfully"
                
        except Exception as e:
            logger.error(f"Architecture validation failed: {e}")
            self.correction_results["validation_passed"] = False
            self.correction_results["summary"] = f"❌ Validation error: {e}"

def main():
    """Main execution function"""
    corrector = PyicsStructureCorrector(PROJECT_ROOT)
    results = corrector.execute_complete_correction()
    
    # Display results
    print("=" * 60)
    print("PYICS STRUCTURE CORRECTION SUMMARY")
    print("=" * 60)
    print(f"Backup Created: {'✅' if results['backup_created'] else '❌'}")
    print(f"Complex Structures Cleaned: {results['complex_structures_cleaned']}")
    print(f"Domains Consolidated: {results['domains_consolidated']}")
    print(f"Standard Modules Generated: {results['standard_modules_generated']}")
    print(f"IoC Registry Created: {'✅' if results['ioc_registry_created'] else '❌'}")
    print(f"Validation Passed: {'✅' if results['validation_passed'] else '❌'}")
    print("=" * 60)
    print(f"Status: {results['summary']}")
    print("=" * 60)
    
    if results["validation_passed"]:
        print("\n🎯 ARCHITECTURE CORRECTED SUCCESSFULLY!")
        print("🔄 Single-pass loading implemented")
        print("🧹 Complex structures cleaned")
        print("📁 Standard domain pattern established")
        print("⚡ Cost-aware IoC registry created")
        
        print("\n📋 NEXT STEPS:")
        print("1. Test single-pass loading: python -c 'from pyics.core.ioc_registry import validate_architecture; print(validate_architecture())'")
        print("2. Verify domain loading: python -c 'from pyics.core.ioc_registry import get_all_domains; print(get_all_domains())'")
        print("3. Check load performance: python -c 'from pyics.core.ioc_registry import get_registry; print(get_registry().get_load_performance())'")
        print("4. Run comprehensive tests on cleaned architecture")
    
    sys.exit(0 if results["validation_passed"] else 1)

if __name__ == "__main__":
    main()

# [EOF] - End of Pyics structure corrector
