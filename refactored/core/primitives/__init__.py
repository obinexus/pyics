#!/usr/bin/env python3
"""
refactored/core/primitives/__init__.py
Primitives Domain - Foundational Atomic Operations and Data Types

Generated: 2025-05-31T18:56:35.568872
Domain: primitives
Responsibility: Thread-safe atomic operations and immutable data types
Compute Weight: 0.1 (minimal computational overhead)
Architecture: Dependency Level 0 - Foundational layer with zero external dependencies
Thread Safety: Guaranteed through atomic operation design and immutable data structures
Deterministic: Yes - all operations are pure functions with mathematical guarantees

Author: OBINexus Engineering Team / Nnamdi Okpala
Architecture: Single-Pass RIFT System - Core Primitives Domain
Phase: 3.1.6.1 - Foundational Structuring
"""

import logging
from typing import Any, Dict

# Import all components from domain modules
from .data_types import (
    AtomicDataType,
    ImmutableContainer,
    AtomicValue, 
    AtomicReference,
    create_atomic_value,
    create_atomic_reference,
    validate_atomic_type,
    get_atomic_type_info
)

from .operations import (
    AtomicOperationError,
    atomic_identity,
    atomic_add,
    atomic_multiply,
    deepcopy_immutable,
    atomic_compose_functions
)

# Domain metadata for cost-aware loading
__domain_metadata__ = {
    "name": "primitives",
    "priority_index": 1,
    "compute_time_weight": 0.1,
    "exposure_type": "core_internal",
    "dependency_level": 0,
    "thread_safe": True,
    "load_order": 1,
    "problem_solved": "Thread-safe atomic operations and immutable data types",
    "modules": ["data_types", "operations"],
    "scaffolding_version": "3.1.6.1",
    "generated": "2025-05-31T18:56:35.568872"
}

logger = logging.getLogger("pyics.core.primitives")

class PrimitivesDomainCoordinator:
    """
    Domain coordinator for primitives with atomic operation guarantees
    
    Manages primitive operations and data types while maintaining
    thread safety and immutability constraints.
    """
    
    def __init__(self):
        self._initialized = False
        self._atomic_operations = {}
        self._data_type_factories = {}
    
    def initialize_domain(self) -> bool:
        """Initialize primitives domain with validation"""
        try:
            # Test core atomic operations
            identity_test = atomic_identity(42)
            if identity_test != 42:
                logger.error("Atomic identity validation failed")
                return False
            
            add_test = atomic_add(2, 3)
            if add_test != 5:
                logger.error("Atomic addition validation failed")
                return False
            
            # Test data type factories
            test_value = create_atomic_value(42)
            if not validate_atomic_type(test_value):
                logger.error("AtomicValue factory validation failed")
                return False
            
            # Register operations and factories
            self._atomic_operations = {
                'identity': atomic_identity,
                'add': atomic_add,
                'multiply': atomic_multiply,
                'deepcopy': deepcopy_immutable,
                'compose': atomic_compose_functions
            }
            
            self._data_type_factories = {
                'atomic_value': create_atomic_value,
                'atomic_reference': create_atomic_reference
            }
            
            self._initialized = True
            logger.info("Primitives domain initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Primitives domain initialization failed: {e}")
            return False
    
    def get_atomic_operations(self) -> Dict[str, Any]:
        """Get all registered atomic operations"""
        if not self._initialized:
            self.initialize_domain()
        return self._atomic_operations.copy()
    
    def get_data_type_factories(self) -> Dict[str, Any]:
        """Get all registered data type factories"""
        if not self._initialized:
            self.initialize_domain()
        return self._data_type_factories.copy()

# Global domain coordinator
_domain_coordinator = PrimitivesDomainCoordinator()

def get_domain_exports() -> Dict[str, Any]:
    """Export all domain capabilities for registration"""
    return {
        # Data types
        'AtomicDataType': AtomicDataType,
        'ImmutableContainer': ImmutableContainer,
        'AtomicValue': AtomicValue,
        'AtomicReference': AtomicReference,
        
        # Data type factories
        'create_atomic_value': create_atomic_value,
        'create_atomic_reference': create_atomic_reference,
        
        # Operations
        'atomic_identity': atomic_identity,
        'atomic_add': atomic_add, 
        'atomic_multiply': atomic_multiply,
        'deepcopy_immutable': deepcopy_immutable,
        'atomic_compose_functions': atomic_compose_functions,
        
        # Validation utilities
        'validate_atomic_type': validate_atomic_type,
        'get_atomic_type_info': get_atomic_type_info,
        
        # Error handling
        'AtomicOperationError': AtomicOperationError,
        
        # Domain coordination
        'domain_coordinator': _domain_coordinator
    }

def get_domain_metadata() -> Dict[str, Any]:
    """Return domain metadata for cost-aware loading"""
    return __domain_metadata__.copy()

def initialize_primitives_domain() -> bool:
    """Initialize primitives domain with comprehensive validation"""
    return _domain_coordinator.initialize_domain()

# Dynamic __all__ export - includes all public components
__all__ = [
    # Protocol and base classes
    'AtomicDataType',
    'ImmutableContainer',
    
    # Concrete data types
    'AtomicValue',
    'AtomicReference',
    
    # Factory functions
    'create_atomic_value',
    'create_atomic_reference',
    
    # Core atomic operations
    'atomic_identity',
    'atomic_add',
    'atomic_multiply',
    'deepcopy_immutable',
    'atomic_compose_functions',
    
    # Validation utilities
    'validate_atomic_type',
    'get_atomic_type_info',
    
    # Error handling
    'AtomicOperationError',
    
    # Domain management
    'get_domain_exports',
    'get_domain_metadata',
    'initialize_primitives_domain',
    'PrimitivesDomainCoordinator'
]

# Auto-initialize domain on module load
if not initialize_primitives_domain():
    raise RuntimeError("Failed to initialize primitives domain")

logger.info(f"Primitives domain loaded with {len(__all__)} exported components")

# [EOF] - End of __init__.py module
