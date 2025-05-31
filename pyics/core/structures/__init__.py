#!/usr/bin/env python3
"""
pyics/core/structures/__init__.py
Structures Domain Module

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: structures
"""

# Import domain configuration
from .config import get_domain_metadata, validate_configuration, cost_metadata

#!/usr/bin/env python3
"""
pyics/core/structures/__init__.py
Structures Domain - Modular ABC Contract Architecture

PROBLEM SOLVED: Immutable data containers and structure definitions
ARCHITECTURE: Single-pass dependency isolation with ABC contract extensions
MODULES: Problem-classified modular segmentation

Author: OBINexus Engineering Team / Nnamdi Okpala
Architecture: Single-Pass RIFT System with Cost-Aware Loading
Phase: 3.1.6.1 - Modular Problem Classification
"""

from typing import Any, Dict, List, Optional
import logging

# Import all modular components
from .enumerations import get_module_exports as enumerations_exports
from .immutable_event import get_module_exports as immutable_event_exports
from .calendar_data import get_module_exports as calendar_data_exports
from .distribution_structures import get_module_exports as distribution_structures_exports
from .audit_structures import get_module_exports as audit_structures_exports

# Domain metadata for cost-aware loading
__domain_metadata__ = {
    "name": "structures",
    "priority_index": 2,
    "compute_time_weight": 0.2,
    "exposure_type": "version_required",
    "dependency_level": 2,
    "thread_safe": True,
    "load_order": 3,
    "modular_restructure": "2025-05-31",
    "module_count": 5
}

logger = logging.getLogger(f"pyics.core.structures")

class StructuresDomainCoordinator:
    """
    Domain coordinator for modular ABC contract management
    
    Manages module registration, dependency resolution, and contract validation
    """
    
    def __init__(self):
        self._modules = {}
        self._contracts = {}
        self._initialized = False
    
    def register_modules(self) -> bool:
        """Register all domain modules with contract validation"""
        try:
            module_exports = {
                'enumerations': enumerations_exports(),
        'immutable_event': immutable_event_exports(),
        'calendar_data': calendar_data_exports(),
        'distribution_structures': distribution_structures_exports(),
        'audit_structures': audit_structures_exports()
            }
            
            for module_name, exports in module_exports.items():
                self._modules[module_name] = exports
                logger.info(f"Registered module: {module_name}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Module registration failed: {e}")
            return False
    
    def get_module_contracts(self) -> Dict[str, Any]:
        """Get all ABC contracts from registered modules"""
        contracts = {}
        
        for module_name, exports in self._modules.items():
            for export_name, export_obj in exports.items():
                if export_name.endswith('Protocol') or export_name.endswith('Base'):
                    contracts[f"{module_name}.{export_name}"] = export_obj
        
        return contracts
    
    def validate_domain_integrity(self) -> bool:
        """Validate domain maintains ABC contract integrity"""
        try:
            if not self._initialized:
                return False
            
            # Validate all modules are properly registered
            expected_modules = {'enumerations', 'audit_structures', 'distribution_structures', 'calendar_data', 'immutable_event'}
            registered_modules = set(self._modules.keys())
            
            if expected_modules != registered_modules:
                logger.error(f"Module registration mismatch: expected {expected_modules}, got {registered_modules}")
                return False
            
            # Validate contract structure
            contracts = self.get_module_contracts()
            if not contracts:
                logger.error("No ABC contracts found in domain")
                return False
            
            logger.info(f"Domain integrity validated: {len(self._modules)} modules, {len(contracts)} contracts")
            return True
            
        except Exception as e:
            logger.error(f"Domain integrity validation failed: {e}")
            return False

# Global domain coordinator instance
_domain_coordinator = StructuresDomainCoordinator()

def get_domain_exports() -> Dict[str, Any]:
    """Export all domain capabilities for registration"""
    if not _domain_coordinator._initialized:
        _domain_coordinator.register_modules()
    
    exports = {}
    
    # Export all module capabilities
    for module_name, module_exports in _domain_coordinator._modules.items():
        for export_name, export_obj in module_exports.items():
            exports[f"{module_name}_{export_name}"] = export_obj
    
    # Export domain coordinator
    exports['domain_coordinator'] = _domain_coordinator
    
    return exports

def get_domain_metadata() -> Dict[str, Any]:
    """Return domain metadata for cost-aware loading"""
    return __domain_metadata__.copy()

def get_module_list() -> List[str]:
    """Return list of all modules in domain"""
    return ['"enumerations"', '"immutable_event"', '"calendar_data"', '"distribution_structures"', '"audit_structures"']

def validate_domain() -> bool:
    """Validate domain follows modular ABC contract architecture"""
    return _domain_coordinator.validate_domain_integrity()

def initialize_domain() -> bool:
    """Initialize domain with modular structure and ABC contracts"""
    try:
        if not _domain_coordinator.register_modules():
            return False
        
        if not _domain_coordinator.validate_domain_integrity():
            return False
        
        logger.info(f"Domain {__domain_metadata__['name']} initialized with {len(get_module_list())} modules")
        return True
        
    except Exception as e:
        logger.error(f"Domain initialization failed: {e}")
        return False

# Export for cost-aware loading
__all__ = [
    'get_domain_exports',
    'get_domain_metadata', 
    'get_module_list',
    'validate_domain',
    'initialize_domain',
    'StructuresDomainCoordinator'
]

# Self-validation on domain load
if not initialize_domain():
    raise RuntimeError(f"Failed to initialize domain: structures")


# Export configuration interfaces
__all__ = getattr(globals(), '__all__', []) + [
    "get_domain_metadata",
    "validate_configuration",
    "cost_metadata"
]

# [EOF] - End of structures domain module
