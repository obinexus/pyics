#!/usr/bin/env python3
"""
pyics/core/validators/__init__.py
Validators Domain - Modular ABC Contract Architecture

PROBLEM SOLVED: Comprehensive validation framework with ABC contracts
ARCHITECTURE: Single-pass dependency isolation with ABC contract extensions
MODULES: Problem-classified modular segmentation

Author: OBINexus Engineering Team / Nnamdi Okpala
Architecture: Single-Pass RIFT System with Cost-Aware Loading
Phase: 3.1.6.1 - Modular Problem Classification
"""

from typing import Any, Dict, List, Optional
import logging

# Import all modular components
from .data_integrity import get_module_exports as data_integrity_exports
from .constraint_validation import get_module_exports as constraint_validation_exports
from .input_validation import get_module_exports as input_validation_exports
from .validation_coordinator import get_module_exports as validation_coordinator_exports
from .validation_errors import get_module_exports as validation_errors_exports

# Domain metadata for cost-aware loading
__domain_metadata__ = {
    "name": "validators",
    "priority_index": 3,
    "compute_time_weight": 0.4,
    "exposure_type": "version_required",
    "dependency_level": 3,
    "thread_safe": True,
    "load_order": 5,
    "modular_restructure": "2025-05-31",
    "module_count": 5
}

logger = logging.getLogger(f"pyics.core.validators")

class ValidatorsDomainCoordinator:
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
                'data_integrity': data_integrity_exports(),
        'constraint_validation': constraint_validation_exports(),
        'input_validation': input_validation_exports(),
        'validation_coordinator': validation_coordinator_exports(),
        'validation_errors': validation_errors_exports()
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
            expected_modules = {'constraint_validation', 'validation_coordinator', 'input_validation', 'validation_errors', 'data_integrity'}
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
_domain_coordinator = ValidatorsDomainCoordinator()

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
    return ['"data_integrity"', '"constraint_validation"', '"input_validation"', '"validation_coordinator"', '"validation_errors"']

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
    'ValidatorsDomainCoordinator'
]

# Self-validation on domain load
if not initialize_domain():
    raise RuntimeError(f"Failed to initialize domain: validators")
