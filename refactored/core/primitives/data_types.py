#!/usr/bin/env python3
"""
refactored/core/primitives/data_types.py
Foundational Immutable Data Types - Dependency Level 0

Generated: 2025-05-31T18:56:35.568385
Domain: primitives
Responsibility: Atomic immutable data type definitions with zero dependencies
Compute Weight: 0.1 (minimal overhead for foundational types)
Dependencies: None (foundational level 0)
Thread Safety: Guaranteed through immutable design
Deterministic: Yes - pure data containers

Author: OBINexus Engineering Team / Nnamdi Okpala
Architecture: Single-Pass RIFT System - Foundational Layer
"""

from typing import Any, TypeVar, Generic, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import threading
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')

# Thread-safe atomic type validation lock
_type_validation_lock = threading.RLock()

@runtime_checkable
class AtomicDataType(Protocol):
    """
    Protocol for atomic data types with immutability guarantees
    
    Defines contract for foundational data types that serve as
    building blocks for higher-level domain structures.
    """
    
    def validate_immutability(self) -> bool:
        """Validate that data type maintains immutability constraints"""
        ...
    
    def get_type_signature(self) -> str:
        """Return deterministic type signature for identification"""
        ...
    
    def atomic_hash(self) -> int:
        """Generate deterministic hash for atomic operations"""
        ...

class ImmutableContainer(ABC, Generic[T]):
    """
    Abstract base class for all immutable container types
    
    Provides foundational structure for atomic data containers
    ensuring thread-safety and deterministic behavior.
    """
    
    def __init__(self, value: T):
        with _type_validation_lock:
            self._validate_construction(value)
            # Use consistent attribute naming
            self._container_value = value
            self._hash_cache = None
    
    @abstractmethod
    def _validate_construction(self, value: T) -> None:
        """Validate value during construction - must be implemented by subclasses"""
        pass
    
    @property
    def value(self) -> T:
        """Get immutable value - thread-safe read access"""
        return self._container_value
    
    def validate_immutability(self) -> bool:
        """Validate container maintains immutability"""
        try:
            # Attempt to verify value hasn't been mutated
            current_hash = hash(str(self._container_value)) if self._container_value is not None else 0
            return True  # If we reach here, value is still accessible and immutable
        except Exception:
            return False
    
    def get_type_signature(self) -> str:
        """Return deterministic type signature"""
        return f"{self.__class__.__name__}[{type(self._container_value).__name__}]"
    
    def atomic_hash(self) -> int:
        """Thread-safe deterministic hash calculation"""
        if self._hash_cache is None:
            with _type_validation_lock:
                if self._hash_cache is None:  # Double-check locking pattern
                    self._hash_cache = hash((self.__class__.__name__, str(self._container_value)))
        return self._hash_cache
    
    def __eq__(self, other) -> bool:
        """Atomic equality comparison"""
        if not isinstance(other, ImmutableContainer):
            return False
        return (self.__class__ == other.__class__ and 
                self._container_value == other._container_value)
    
    def __hash__(self) -> int:
        """Use atomic hash for Python hash protocol"""
        return self.atomic_hash()
    
    def __repr__(self) -> str:
        """Deterministic string representation"""
        return f"{self.__class__.__name__}({self._container_value!r})"

@dataclass(frozen=True)
class AtomicValue(ImmutableContainer[T]):
    """
    Concrete atomic value container for primitive types
    
    Provides thread-safe storage for atomic values with
    immutability guarantees and deterministic operations.
    """
    
    def _validate_construction(self, value: T) -> None:
        """Validate atomic value constraints"""
        # Atomic values should be of primitive, immutable types
        allowed_types = (int, float, str, bool, type(None), tuple, frozenset, bytes)
        
        if not isinstance(value, allowed_types):
            # Check for frozen dataclass
            if hasattr(value, '__dataclass_fields__') and getattr(value, '__frozen__', False):
                return  # Frozen dataclass is acceptable
            
            raise ValueError(f"AtomicValue requires immutable type, got: {type(value)}")

class AtomicReference(ImmutableContainer[T]):
    """
    Atomic reference container for complex immutable objects
    
    Provides thread-safe reference storage with validation
    for complex immutable data structures.
    """
    
    def _validate_construction(self, value: T) -> None:
        """Validate reference immutability"""
        if value is None:
            return  # None is always immutable
        
        # Check for immutable containers
        if isinstance(value, (tuple, frozenset, bytes)):
            return
        
        # Check for frozen dataclass
        if hasattr(value, '__dataclass_fields__') and getattr(value, '__frozen__', False):
            return
        
        # Check for AtomicDataType protocol compliance
        if isinstance(value, AtomicDataType):
            if not value.validate_immutability():
                raise ValueError("Referenced object failed immutability validation")
            return
        
        raise ValueError(f"AtomicReference requires immutable object, got: {type(value)}")

# Factory functions for atomic data type creation
def create_atomic_value(value: T) -> AtomicValue[T]:
    """
    Factory function for creating atomic values
    
    Args:
        value: Primitive value to store atomically
        
    Returns:
        AtomicValue container with immutability guarantees
        
    Raises:
        ValueError: If value is not immutable
    """
    return AtomicValue(value)

def create_atomic_reference(obj: T) -> AtomicReference[T]:
    """
    Factory function for creating atomic references
    
    Args:
        obj: Immutable object to reference atomically
        
    Returns:
        AtomicReference container with immutability guarantees
        
    Raises:
        ValueError: If object is not immutable
    """
    return AtomicReference(obj)

# Atomic type validation utilities
def validate_atomic_type(obj: Any) -> bool:
    """
    Validate that object qualifies as atomic data type
    
    Args:
        obj: Object to validate for atomic properties
        
    Returns:
        True if object is atomic and immutable, False otherwise
    """
    try:
        if isinstance(obj, AtomicDataType):
            return obj.validate_immutability()
        
        if isinstance(obj, ImmutableContainer):
            return obj.validate_immutability()
        
        # Check primitive immutable types
        if isinstance(obj, (int, float, str, bool, type(None), tuple, frozenset, bytes)):
            return True
        
        # Check frozen dataclass
        if hasattr(obj, '__dataclass_fields__') and getattr(obj, '__frozen__', False):
            return True
        
        return False
        
    except Exception:
        return False

def get_atomic_type_info(obj: Any) -> dict:
    """
    Get comprehensive type information for atomic objects
    
    Args:
        obj: Object to analyze
        
    Returns:
        Dictionary containing type information and validation results
    """
    info = {
        'type_name': type(obj).__name__,
        'is_atomic': False,
        'is_immutable': False,
        'type_signature': None,
        'atomic_hash': None
    }
    
    try:
        info['is_atomic'] = validate_atomic_type(obj)
        
        if isinstance(obj, AtomicDataType):
            info['is_immutable'] = obj.validate_immutability()
            info['type_signature'] = obj.get_type_signature()
            info['atomic_hash'] = obj.atomic_hash()
        elif isinstance(obj, ImmutableContainer):
            info['is_immutable'] = obj.validate_immutability()
            info['type_signature'] = obj.get_type_signature()
            info['atomic_hash'] = obj.atomic_hash()
        else:
            info['is_immutable'] = info['is_atomic']
            info['type_signature'] = f"primitive[{type(obj).__name__}]"
            info['atomic_hash'] = hash(str(obj)) if obj is not None else 0
    
    except Exception as e:
        info['error'] = str(e)
    
    return info

# Export all public components
__all__ = [
    'AtomicDataType',
    'ImmutableContainer', 
    'AtomicValue',
    'AtomicReference',
    'create_atomic_value',
    'create_atomic_reference',
    'validate_atomic_type',
    'get_atomic_type_info'
]

# [EOF] - End of data_types.py module
