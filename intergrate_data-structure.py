#!/usr/bin/env python3
"""
integrate_data_structures.py  
Pyics Core Modular Architecture - Data Structure Integration

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Single-pass integration of data structures across core domains
Architecture: DOP-compliant structure registry for dependency resolution
Methodology: Waterfall with systematic structure analysis

Implements systematic discovery and registration of data structures across
all core domains following single-pass RIFT principles.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
import logging

# Configuration
CORE_DIRECTORY = "pyics/core"
REFACTORED_CORE_DIRECTORY = "refactored/core"
REGISTRY_OUTPUT = "pyics/core/registry/structure_registry.py"
LOG_FILE = f"structure_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StructureAnalyzer:
    """
    AST-based analyzer for Python data structures
    
    Extracts dataclasses, enums, protocols, and type annotations
    following DOP principles for systematic structure classification.
    """
    
    def __init__(self):
        self.structures: Dict[str, Dict[str, Any]] = {}
        self.current_domain = ""
        self.current_file = ""
        
    def analyze_file(self, file_path: Path, domain: str) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a single data_types.py file for structure definitions
        
        Args:
            file_path: Path to data_types.py file
            domain: Domain name for structure classification
            
        Returns:
            Dictionary of discovered structures with metadata
        """
        self.current_domain = domain
        self.current_file = str(file_path.relative_to(Path.cwd()))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract structures using AST visitor
            extractor = StructureExtractor(domain, self.current_file)
            extractor.visit(tree)
            
            return extractor.structures
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return {}

class StructureExtractor(ast.NodeVisitor):
    """
    AST visitor for extracting data structure definitions
    
    Identifies and classifies:
    - @dataclass decorated classes
    - Enum definitions  
    - Protocol definitions
    - Type annotations and field definitions
    """
    
    def __init__(self, domain: str, file_path: str):
        self.domain = domain
        self.file_path = file_path
        self.structures: Dict[str, Dict[str, Any]] = {}
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to identify data structures"""
        
        # Analyze class decorators
        is_dataclass = self._has_dataclass_decorator(node)
        is_enum = self._inherits_from_enum(node)
        is_protocol = self._inherits_from_protocol(node)
        
        if is_dataclass or is_enum or is_protocol:
            structure_info = self._extract_structure_info(node, is_dataclass, is_enum, is_protocol)
            self.structures[node.name] = structure_info
            
        self.generic_visit(node)
    
    def _has_dataclass_decorator(self, node: ast.ClassDef) -> bool:
        """Check if class has @dataclass decorator"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
                return True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == 'dataclass':
                    return True
        return False
    
    def _inherits_from_enum(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from Enum"""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ['Enum', 'IntEnum', 'Flag', 'IntFlag']:
                return True
        return False
    
    def _inherits_from_protocol(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from Protocol"""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'Protocol':
                return True
            elif isinstance(base, ast.Subscript):
                if isinstance(base.value, ast.Name) and base.value.id == 'Protocol':
                    return True
        return False
    
    def _extract_structure_info(self, node: ast.ClassDef, is_dataclass: bool, 
                              is_enum: bool, is_protocol: bool) -> Dict[str, Any]:
        """Extract comprehensive structure information"""
        
        # Determine structure type
        if is_dataclass:
            structure_type = "dataclass"
            compute_type = self._determine_compute_type(node)
        elif is_enum:
            structure_type = "enum"
            compute_type = "static"
        elif is_protocol:
            structure_type = "protocol"
            compute_type = "interface"
        else:
            structure_type = "class"
            compute_type = "dynamic"
        
        # Extract field information
        value_types = self._extract_field_types(node, is_dataclass, is_enum)
        
        # Build structure metadata
        structure_info = {
            "domain": self.domain,
            "file": self.file_path,
            "structure_type": structure_type,
            "value_types": value_types,
            "compute_type": compute_type,
            "base_classes": [self._get_base_name(base) for base in node.bases],
            "is_generic": self._is_generic_class(node),
            "docstring": ast.get_docstring(node) or "",
            "line_number": node.lineno
        }
        
        return structure_info
    
    def _determine_compute_type(self, node: ast.ClassDef) -> str:
        """Determine computation type based on class analysis"""
        
        # Check for computed properties or methods
        has_properties = False
        has_methods = False
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if any(isinstance(d, ast.Name) and d.id == 'property' for d in item.decorator_list):
                    has_properties = True
                elif not item.name.startswith('_'):
                    has_methods = True
        
        if has_properties:
            return "computed"
        elif has_methods:
            return "dynamic"
        else:
            return "static"
    
    def _extract_field_types(self, node: ast.ClassDef, is_dataclass: bool, is_enum: bool) -> Dict[str, str]:
        """Extract field type annotations"""
        field_types = {}
        
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                # Handle annotated assignments (field: Type = value)
                field_name = item.target.id
                field_type = self._get_type_annotation(item.annotation)
                field_types[field_name] = field_type
                
            elif isinstance(item, ast.Assign) and is_enum:
                # Handle enum values (FIELD = value)
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        field_types[target.id] = "enum_value"
        
        return field_types
    
    def _get_type_annotation(self, annotation: ast.AST) -> str:
        """Convert AST type annotation to string representation"""
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Subscript):
                if isinstance(annotation.value, ast.Name):
                    base_type = annotation.value.id
                    # Handle generic types like List[str], Dict[str, int]
                    return f"{base_type}[...]"
                return "Generic[...]"
            elif isinstance(annotation, ast.Attribute):
                return f"{self._get_type_annotation(annotation.value)}.{annotation.attr}"
            elif isinstance(annotation, ast.Constant):
                return str(annotation.value)
            else:
                return "Any"
        except Exception:
            return "Any"
    
    def _get_base_name(self, base: ast.AST) -> str:
        """Get base class name from AST node"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self._get_base_name(base.value)}.{base.attr}"
        elif isinstance(base, ast.Subscript):
            return self._get_base_name(base.value)
        else:
            return "Unknown"
    
    def _is_generic_class(self, node: ast.ClassDef) -> bool:
        """Check if class is generic (has type parameters)"""
        for base in node.bases:
            if isinstance(base, ast.Subscript):
                return True
        return False

class StructureIntegrator:
    """
    Main integration coordinator for data structure discovery and registry generation
    
    Implements single-pass architecture for systematic structure analysis
    across all core domains.
    """
    
    def __init__(self):
        self.analyzer = StructureAnalyzer()
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.domains_discovered: Set[str] = set()
        
    def discover_data_types_files(self) -> List[Tuple[Path, str]]:
        """
        Discover all data_types.py files in core domains
        
        Returns:
            List of (file_path, domain_name) tuples
        """
        data_types_files = []
        
        # Check both core directories
        core_paths = [Path(CORE_DIRECTORY), Path(REFACTORED_CORE_DIRECTORY)]
        
        for core_path in core_paths:
            if not core_path.exists():
                logger.info(f"Core directory not found: {core_path}")
                continue
            
            # Walk through core directory structure
            for root, dirs, files in os.walk(core_path):
                if 'data_types.py' in files:
                    file_path = Path(root) / 'data_types.py'
                    
                    # Extract domain name from path
                    relative_path = Path(root).relative_to(core_path)
                    domain_name = str(relative_path) if relative_path != Path('.') else 'core'
                    
                    data_types_files.append((file_path, domain_name))
                    logger.info(f"Discovered data_types.py in domain: {domain_name}")
        
        return data_types_files
    
    def integrate_structures(self) -> Dict[str, Dict[str, Any]]:
        """
        Execute single-pass integration of all data structures
        
        Returns:
            Complete structure registry mapping
        """
        logger.info("Starting single-pass data structure integration...")
        
        # Phase 1: Discover all data_types.py files
        data_types_files = self.discover_data_types_files()
        
        if not data_types_files:
            logger.warning("No data_types.py files found in core domains")
            return {}
        
        # Phase 2: Analyze each file for structure definitions
        for file_path, domain in data_types_files:
            logger.info(f"Analyzing structures in {domain}: {file_path}")
            
            domain_structures = self.analyzer.analyze_file(file_path, domain)
            
            # Merge into global registry
            for structure_name, structure_info in domain_structures.items():
                if structure_name in self.registry:
                    logger.warning(f"Structure name collision: {structure_name} in {domain}")
                    # Namespace collision handling
                    namespaced_name = f"{domain}.{structure_name}"
                    self.registry[namespaced_name] = structure_info
                else:
                    self.registry[structure_name] = structure_info
                
            self.domains_discovered.add(domain)
        
        logger.info(f"Integration complete: {len(self.registry)} structures from {len(self.domains_discovered)} domains")
        return self.registry
    
    def generate_registry_file(self) -> bool:
        """
        Generate structure_registry.py with complete structure mapping
        
        Returns:
            True if generation successful, False otherwise
        """
        try:
            # Ensure registry directory exists
            registry_dir = Path(REGISTRY_OUTPUT).parent
            registry_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate registry content
            registry_content = self._build_registry_content()
            
            # Write registry file
            with open(REGISTRY_OUTPUT, 'w', encoding='utf-8') as f:
                f.write(registry_content)
            
            logger.info(f"Structure registry generated: {REGISTRY_OUTPUT}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate registry file: {e}")
            return False
    
    def _build_registry_content(self) -> str:
        """Build complete registry file content"""
        
        timestamp = datetime.now().isoformat()
        
        content = f'''#!/usr/bin/env python3
"""
pyics/core/registry/structure_registry.py
Pyics Core Data Structure Registry - Auto-Generated

Generated: {timestamp}
Engineering Lead: Nnamdi Okpala / OBINexus Computing
Architecture: Single-Pass RIFT System - Structure Integration Registry

PROBLEM SOLVED: Centralized structure discovery and dependency resolution
DEPENDENCIES: None (generated registry with static mappings)
THREAD SAFETY: Yes - immutable registry data
DETERMINISTIC: Yes - static structure mappings

This registry provides systematic access to all data structures across
core domains, enabling dependency resolution and architectural reflection.
"""

from typing import Dict, Any, Set, List
import logging

logger = logging.getLogger("pyics.core.registry.structure_registry")

# Complete structure registry mapping
STRUCTURE_REGISTRY: Dict[str, Dict[str, Any]] = {{
'''
        
        # Add structure entries
        for structure_name, structure_info in sorted(self.registry.items()):
            content += f'    "{structure_name}": {{\n'
            for key, value in structure_info.items():
                if isinstance(value, str):
                    content += f'        "{key}": "{value}",\n'
                elif isinstance(value, dict):
                    content += f'        "{key}": {value},\n'
                elif isinstance(value, list):
                    content += f'        "{key}": {value},\n'
                else:
                    content += f'        "{key}": {repr(value)},\n'
            content += '    },\n'
        
        content += '''}

# Domain mapping for systematic access
DOMAIN_STRUCTURE_MAP: Dict[str, List[str]] = {
'''
        
        # Build domain mapping
        domain_map = {}
        for structure_name, structure_info in self.registry.items():
            domain = structure_info['domain']
            if domain not in domain_map:
                domain_map[domain] = []
            domain_map[domain].append(structure_name)
        
        for domain, structures in sorted(domain_map.items()):
            content += f'    "{domain}": {structures},\n'
        
        content += f'''}}

# Registry metadata
REGISTRY_METADATA = {{
    "generated": "{timestamp}",
    "total_structures": {len(self.registry)},
    "domains_covered": {sorted(list(self.domains_discovered))},
    "structure_types": {self._get_structure_type_summary()},
    "generation_log": "{LOG_FILE}"
}}

def validate_structure_registry() -> bool:
    """
    Validate structure registry for completeness and consistency
    
    Returns:
        True if registry is valid, False if issues detected
    """
    validation_errors = []
    
    for structure_name, structure_info in STRUCTURE_REGISTRY.items():
        # Validate required fields
        required_fields = ["domain", "file", "structure_type", "compute_type"]
        for field in required_fields:
            if field not in structure_info:
                validation_errors.append(f"Missing {{field}} in {{structure_name}}")
        
        # Warn about empty structures
        value_types = structure_info.get("value_types", {{}})
        if not value_types and structure_info.get("structure_type") == "dataclass":
            logger.warning(f"Empty dataclass detected: {{structure_name}}")
    
    if validation_errors:
        logger.error(f"Registry validation failed with {{len(validation_errors)}} errors")
        for error in validation_errors:
            logger.error(f"  - {{error}}")
        return False
    
    logger.info(f"Registry validation passed: {{len(STRUCTURE_REGISTRY)}} structures")
    return True

def get_structure_info(structure_name: str) -> Dict[str, Any]:
    """Get structure information by name"""
    return STRUCTURE_REGISTRY.get(structure_name, {{}})

def get_structures_by_domain(domain: str) -> List[str]:
    """Get all structures for a domain"""
    return DOMAIN_STRUCTURE_MAP.get(domain, [])

def get_structures_by_type(structure_type: str) -> List[str]:
    """Get structures by type"""
    return [
        name for name, info in STRUCTURE_REGISTRY.items()
        if info.get("structure_type") == structure_type
    ]

# Export registry interface
__all__ = [
    'STRUCTURE_REGISTRY',
    'DOMAIN_STRUCTURE_MAP', 
    'REGISTRY_METADATA',
    'validate_structure_registry',
    'get_structure_info',
    'get_structures_by_domain',
    'get_structures_by_type'
]

# Auto-validate registry on module load
if not validate_structure_registry():
    logger.warning("Structure registry loaded with validation warnings")
else:
    logger.info(f"Structure registry loaded successfully: {{REGISTRY_METADATA['total_structures']}} structures")

# [EOF] - End of structure_registry.py module
'''
        
        return content
    
    def _get_structure_type_summary(self) -> Dict[str, int]:
        """Get summary of structure types in registry"""
        type_counts = {}
        for structure_info in self.registry.values():
            structure_type = structure_info.get("structure_type", "unknown")
            type_counts[structure_type] = type_counts.get(structure_type, 0) + 1
        return type_counts

def main():
    """Main execution function for structure integration"""
    
    logger.info("=" * 60)
    logger.info("Pyics Data Structure Integration - Single-Pass Architecture")
    logger.info("Engineering Lead: Nnamdi Okpala / OBINexus Computing")
    logger.info("=" * 60)
    
    try:
        # Initialize integrator
        integrator = StructureIntegrator()
        
        # Execute single-pass integration
        registry = integrator.integrate_structures()
        
        if not registry:
            logger.error("No structures discovered - integration failed")
            sys.exit(1)
        
        # Generate registry file
        if not integrator.generate_registry_file():
            logger.error("Registry file generation failed")
            sys.exit(1)
        
        # Summary report
        logger.info("=" * 60)
        logger.info("INTEGRATION SUMMARY")
        logger.info(f"✅ Structures Discovered: {len(registry)}")
        logger.info(f"✅ Domains Processed: {len(integrator.domains_discovered)}")
        logger.info(f"✅ Registry Generated: {REGISTRY_OUTPUT}")
        logger.info(f"✅ Log File: {LOG_FILE}")
        logger.info("=" * 60)
        
        # Structure type breakdown
        type_summary = integrator._get_structure_type_summary()
        logger.info("Structure Type Breakdown:")
        for struct_type, count in type_summary.items():
            logger.info(f"  {struct_type}: {count}")
        
        logger.info("🔄 Next Phase: Registry validation and domain integration testing")
        
    except KeyboardInterrupt:
        logger.info("Integration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Integration failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# [EOF] - End of integrate_data_structures.py
