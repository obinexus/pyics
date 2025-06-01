#!/usr/bin/env python3
"""
pyics/core/primitives/implementations/atomic_operations.py
Linear Primitives - Atomic Operations (Dependency Level 0)

Contains fundamental atomic operations with NO external dependencies.
These operations form the foundation of all higher-level compositions.

SAFETY CONSTRAINTS:
- NO external imports beyond Python standard library
- ALL operations must be thread-safe and immutable
- NO state mutation allowed
- Execution time must be deterministic and bounded
"""

from typing import Any, TypeVar, Callable, Optional
from functools import wraps
import threading

T = TypeVar('T')
U = TypeVar('U')

# Thread-safe atomic operations
_operation_lock = threading.RLock()

def atomic_identity(value: T) -> T:
    """
    Atomic identity operation - thread-safe foundation primitive
    
    Args:
        value: Input value of any type
        
    Returns:
        Exact same value (guaranteed no mutation)
        
    Thread Safety: Yes - no shared state accessed
    Deterministic: Yes - O(1) execution time
    """
    return value

def atomic_validate_immutable(value: Any) -> bool:
    """
    Validate that value represents immutable data
    
    Args:
        value: Value to validate for immutability
        
    Returns:
        True if value is immutable, False otherwise
        
    Thread Safety: Yes - read-only operation
    """
    immutable_types = (int, float, str, bool, tuple, frozenset, type(None))
    
    if isinstance(value, immutable_types):
        return True
    
    # Check for frozen dataclass
    if hasattr(value, '__dataclass_fields__') and hasattr(value, '__frozen__'):
        return getattr(value, '__frozen__', False)
    
    return False

def atomic_compose_two(f: Callable[[U], T], g: Callable[[Any], U]) -> Callable[[Any], T]:
    """
    Atomic composition of exactly two functions - thread-safe primitive
    
    Args:
        f: Second function to apply
        g: First function to apply
        
    Returns:
        Composed function that applies g then f
        
    Thread Safety: Yes - functional composition creates new function
    """
    @wraps(f)
    def composed(*args, **kwargs):
        with _operation_lock:
            intermediate = g(*args, **kwargs)
            return f(intermediate)
    return composed

def atomic_type_check(value: Any, expected_type: type) -> bool:
    """
    Thread-safe type validation
    
    Args:
        value: Value to check
        expected_type: Expected type
        
    Returns:
        True if value matches expected type
        
    Thread Safety: Yes - read-only type inspection
    """
    return isinstance(value, expected_type)

# Export atomic primitives
def get_domain_exports():
    """Export all atomic operations for registration"""
    return {
        'atomic_identity': atomic_identity,
        'atomic_validate_immutable': atomic_validate_immutable,
        'atomic_compose_two': atomic_compose_two,
        'atomic_type_check': atomic_type_check,
    }

# Validation of primitive module integrity
def validate_primitives() -> bool:
    """Validate all primitive operations maintain atomic properties"""
    test_value = "test"
    
    # Test identity preservation
    if atomic_identity(test_value) != test_value:
        return False
    
    # Test immutability validation
    if not atomic_validate_immutable(test_value):
        return False
    
    # Test composition
    def add_one(x): return x + 1
    def multiply_two(x): return x * 2
    
    composed = atomic_compose_two(multiply_two, add_one)
    if composed(3) != 8:  # (3 + 1) * 2 = 8
        return False
    
    return True

# Self-validation on module load
if not validate_primitives():
    raise RuntimeError("Primitive operations validation failed")
