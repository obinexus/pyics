#!/usr/bin/env python3
"""
pyics/core/composition/__init__.py
Composition Domain - Modular ABC Contract Architecture

PROBLEM SOLVED: Lambda calculus and function composition operations
ARCHITECTURE: Single-pass dependency isolation with ABC contract extensions
MODULES: Problem-classified modular segmentation

Author: OBINexus Engineering Team / Nnamdi Okpala
Architecture: Single-Pass RIFT System with Cost-Aware Loading
Phase: 3.1.6.1 - Modular Problem Classification
"""

from typing import Any, Dict, List, Optional
import logging

# Import all modular components
from .lambda_calculus import get_module_exports as lambda_calculus_exports
from .function_composition import get_module_exports as function_composition_exports
from .composition_engine import get_module_exports as composition_engine_exports
from .mathematical_composition import get_module_exports as mathematical_composition_exports

# Domain metadata for cost-aware loading
__domain_metadata__ = {
    "name": "composition",
    "priority_index": 2,
    "compute_time_weight": 0.3,
    "exposure_type": "version_required",
    "dependency_level": 2,
    "thread_safe": True,
    "load_order": 4,
    "modular_restructure": "2025-05-31",
    "module_count": 4
}

logger = logging.getLogger(f"pyics.core.composition")

class CompositionDomainCoordinator:
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
                'lambda_calculus': lambda_calculus_exports(),
        'function_composition': function_composition_exports(),
        'composition_engine': composition_engine_exports(),
        'mathematical_composition': mathematical_composition_exports()
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
            expected_modules = {'lambda_calculus', 'mathematical_composition', 'composition_engine', 'function_composition'}
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
_domain_coordinator = CompositionDomainCoordinator()

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
    return ['"lambda_calculus"', '"function_composition"', '"composition_engine"', '"mathematical_composition"']

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
    'CompositionDomainCoordinator'
]

# Self-validation on domain load
if not initialize_domain():
    raise RuntimeError(f"Failed to initialize domain: composition")
