#!/usr/bin/env python3
"""
refactored/core/primitives/operations.py
Atomic Operations - Pure Functions with Zero Dependencies

Generated: 2025-05-31T18:56:35.568616
Domain: primitives  
Responsibility: Thread-safe atomic operations with mathematical guarantees
Compute Weight: 0.1 (minimal overhead for atomic operations)
Dependencies: Only data_types.py within same domain
Thread Safety: Guaranteed through pure functional design
Deterministic: Yes - all operations are pure functions

Author: OBINexus Engineering Team / Nnamdi Okpala
Architecture: Single-Pass RIFT System - Foundational Operations Layer
"""

from typing import Any, TypeVar, Callable, Optional, Tuple, List
import threading
import copy
from .data_types import AtomicDataType, ImmutableContainer, validate_atomic_type

T = TypeVar('T')
U = TypeVar('U')

# Thread-safe operation coordination lock
_operation_lock = threading.RLock()

class AtomicOperationError(Exception):
    """Raised when atomic operation constraints are violated"""
    pass

# ===== CORE ATOMIC OPERATIONS =====

def atomic_identity(value: T) -> T:
    """
    Atomic identity operation - fundamental mathematical primitive
    
    Mathematical basis: λx.x (identity function in lambda calculus)
    
    Args:
        value: Input value of any type
        
    Returns:
        Exact same value unchanged (guaranteed no mutation)
        
    Thread Safety: Yes - no shared state modification
    Deterministic: Yes - O(1) execution time
    """
    return value

def atomic_add(a: Any, b: Any) -> Any:
    """
    Thread-safe atomic addition operation
    
    Performs addition while maintaining atomic operation guarantees.
    Supports numeric types and compatible container types.
    
    Args:
        a: First operand
        b: Second operand
        
    Returns:
        Result of addition operation
        
    Raises:
        AtomicOperationError: If operands don't support addition
        
    Thread Safety: Yes - pure function with no shared state
    """
    try:
        with _operation_lock:
            # Validate operands are appropriate for addition
            if not isinstance(a, (int, float, str, tuple, list)):
                if not hasattr(a, '__add__'):
                    raise AtomicOperationError(f"Operand type {type(a)} doesn't support addition")
            
            result = a + b
            
            # Ensure result maintains atomicity if possible
            if validate_atomic_type(result):
                return result
            else:
                # Convert to atomic type if possible
                if isinstance(result, (int, float, str, tuple)):
                    return result  # These are inherently atomic
                else:
                    raise AtomicOperationError(f"Addition result type {type(result)} is not atomic")
                    
    except TypeError as e:
        raise AtomicOperationError(f"Type error in atomic addition: {e}")
    except Exception as e:
        raise AtomicOperationError(f"Unexpected error in atomic addition: {e}")

def atomic_multiply(a: Any, b: Any) -> Any:
    """
    Thread-safe atomic multiplication operation
    
    Args:
        a: First operand  
        b: Second operand
        
    Returns:
        Result of multiplication operation
        
    Thread Safety: Yes - pure function
    """
    try:
        with _operation_lock:
            if not isinstance(a, (int, float)):
                if not hasattr(a, '__mul__'):
                    raise AtomicOperationError(f"Operand type {type(a)} doesn't support multiplication")
            
            result = a * b
            
            if validate_atomic_type(result):
                return result
            else:
                if isinstance(result, (int, float, str, tuple)):
                    return result
                else:
                    raise AtomicOperationError(f"Multiplication result type {type(result)} is not atomic")
                    
    except Exception as e:
        raise AtomicOperationError(f"Error in atomic multiplication: {e}")

def deepcopy_immutable(obj: T) -> T:
    """
    Create deep copy of object while maintaining immutability constraints
    
    Performs deep copy operation with validation that the result
    maintains immutability guarantees required for atomic operations.
    
    Args:
        obj: Object to copy (must be immutable)
        
    Returns:
        Deep copy of object with immutability preserved
        
    Raises:
        AtomicOperationError: If object is mutable or copy fails validation
        
    Thread Safety: Yes - creates new objects without shared state
    Deterministic: Yes - pure functional operation
    """
    try:
        with _operation_lock:
            # Validate source object immutability
            if not validate_atomic_type(obj):
                raise AtomicOperationError(f"Cannot deep copy mutable object: {type(obj)}")
            
            # For inherently immutable types, return as-is (optimization)
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            
            # Handle immutable containers
            if isinstance(obj, tuple):
                return tuple(deepcopy_immutable(item) for item in obj)
            elif isinstance(obj, frozenset):
                return frozenset(deepcopy_immutable(item) for item in obj)
            
            # Handle AtomicDataType protocol objects
            if isinstance(obj, AtomicDataType):
                try:
                    # Use Python's deepcopy for complex atomic objects
                    copied = copy.deepcopy(obj)
                    
                    # Validate copy maintains immutability
                    if not copied.validate_immutability():
                        raise AtomicOperationError("Deep copy failed immutability validation")
                    
                    return copied
                except Exception as e:
                    raise AtomicOperationError(f"Failed to deep copy atomic object: {e}")
            
            # Handle ImmutableContainer objects
            if isinstance(obj, ImmutableContainer):
                try:
                    copied = copy.deepcopy(obj)
                    if not copied.validate_immutability():
                        raise AtomicOperationError("Container copy failed immutability validation")
                    return copied
                except Exception as e:
                    raise AtomicOperationError(f"Failed to deep copy immutable container: {e}")
            
            # Handle frozen dataclasses
            if hasattr(obj, '__dataclass_fields__') and getattr(obj, '__frozen__', False):
                try:
                    return copy.deepcopy(obj)
                except Exception as e:
                    raise AtomicOperationError(f"Failed to deep copy frozen dataclass: {e}")
            
            # If we reach here, object passed validation but isn't handled
            raise AtomicOperationError(f"Unsupported immutable type for deep copy: {type(obj)}")
            
    except AtomicOperationError:
        raise  # Re-raise atomic operation errors
    except Exception as e:
        raise AtomicOperationError(f"Unexpected error in deepcopy_immutable: {e}")

def atomic_compose_functions(f: Callable[[U], T], g: Callable[[Any], U]) -> Callable[[Any], T]:
    """
    Atomic function composition - mathematical foundation operation
    
    Mathematical basis: (f ∘ g)(x) = f(g(x))
    
    Args:
        f: Second function to apply
        g: First function to apply
        
    Returns:
        Composed function that applies g then f atomically
        
    Thread Safety: Yes - creates new function without shared state
    Deterministic: Yes - mathematical function composition
    """
    def composed_function(*args, **kwargs):
        with _operation_lock:
            try:
                intermediate_result = g(*args, **kwargs)
                final_result = f(intermediate_result)
                return final_result
            except Exception as e:
                raise AtomicOperationError(f"Error in composed function execution: {e}")
    
    return composed_function

# Export all atomic operations
__all__ = [
    'AtomicOperationError',
    'atomic_identity',
    'atomic_add',
    'atomic_multiply', 
    'deepcopy_immutable',
    'atomic_compose_functions'
]

# [EOF] - End of operations.py module
