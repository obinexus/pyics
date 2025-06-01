#!/usr/bin/env python3
"""
pyics/core/registry/__init__.py
Registry Domain Module

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: registry
"""

# Import domain configuration
from .config import get_domain_metadata, validate_configuration, cost_metadata

#!/usr/bin/env python3
"""
pyics/core/registry/__init__.py
Registry Domain - Modular ABC Contract Architecture

PROBLEM SOLVED: Component registration and discovery services
ARCHITECTURE: Single-pass dependency isolation with ABC contract extensions
MODULES: Problem-classified modular segmentation

Author: OBINexus Engineering Team / Nnamdi Okpala
Architecture: Single-Pass RIFT System with Cost-Aware Loading
Phase: 3.1.6.1 - Modular Problem Classification
"""

from typing import Any, Dict, List, Optional
import logging

# Import all modular components
from .component_registry import get_module_exports as component_registry_exports
from .discovery_service import get_module_exports as discovery_service_exports
from .lifecycle_management import get_module_exports as lifecycle_management_exports
from .registration_contracts import get_module_exports as registration_contracts_exports

# Domain metadata for cost-aware loading
__domain_metadata__ = {
    "name": "registry",
    "priority_index": 5,
    "compute_time_weight": 0.5,
    "exposure_type": "version_required",
    "dependency_level": 5,
    "thread_safe": True,
    "load_order": 7,
    "modular_restructure": "2025-05-31",
    "module_count": 4
}

logger = logging.getLogger(f"pyics.core.registry")

class RegistryDomainCoordinator:
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
                'component_registry': component_registry_exports(),
        'discovery_service': discovery_service_exports(),
        'lifecycle_management': lifecycle_management_exports(),
        'registration_contracts': registration_contracts_exports()
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
            expected_modules = {'component_registry', 'discovery_service', 'registration_contracts', 'lifecycle_management'}
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
_domain_coordinator = RegistryDomainCoordinator()

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
    return ['"component_registry"', '"discovery_service"', '"lifecycle_management"', '"registration_contracts"']

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
    'RegistryDomainCoordinator'
]

# Self-validation on domain load
if not initialize_domain():
    raise RuntimeError(f"Failed to initialize domain: registry")


# Export configuration interfaces
__all__ = getattr(globals(), '__all__', []) + [
    "get_domain_metadata",
    "validate_configuration",
    "cost_metadata"
]

# [EOF] - End of registry domain module
