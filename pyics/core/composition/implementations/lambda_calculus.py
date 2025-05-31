#!/usr/bin/env python3
"""
pyics/core/composition/implementations/lambda_calculus.py
Linear Composition - Lambda Calculus Operations (Dependency Level 2)

Implements lambda calculus operations following linear single-pass architecture.
All composition operations maintain thread-safety and immutability.

DEPENDENCIES:
- primitives.atomic_operations (level 0)
- protocols.linear_interfaces (level 1)
"""

from typing import Callable, TypeVar, Any, List, Optional
from functools import reduce, wraps
import threading

# Import only from lower dependency levels
from ...primitives.implementations.atomic_operations import (
    atomic_identity, atomic_compose_two, atomic_validate_immutable
)
from ...protocols.implementations.linear_interfaces import (
    Composable, ValidationError
)

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

# Thread-safe composition operations
_composition_lock = threading.RLock()

class LinearComposition(Composable):
    """Thread-safe function composition following linear principles"""
    
    def __init__(self, *functions: Callable):
        if not functions:
            self._composition = atomic_identity
        elif len(functions) == 1:
            self._composition = functions[0]
        else:
            self._composition = self._create_composition(*functions)
        
        self._function_count = len(functions)
        self._validated = False
    
    def _create_composition(self, *functions: Callable) -> Callable:
        """Create composition using atomic operations only"""
        def compose_reducer(f: Callable, g: Callable) -> Callable:
            return atomic_compose_two(f, g)
        
        return reduce(compose_reducer, functions)
    
    def compose_with(self, other: 'Composable') -> 'Composable':
        """Compose with another composable maintaining single-pass"""
        if not isinstance(other, LinearComposition):
            raise ValidationError(
                "composition_type_error",
                "Can only compose with other LinearComposition instances"
            )
        
        new_composition = atomic_compose_two(
            self._composition, 
            other._composition
        )
        
        result = LinearComposition()
        result._composition = new_composition
        result._function_count = self._function_count + other._function_count
        
        return result
    
    def validate_purity(self) -> bool:
        """Validate composition maintains purity constraints"""
        if self._validated:
            return True
        
        # Validate composition chain maintains single-pass property
        try:
            # Test with immutable input
            test_input = "linear_test"
            if not atomic_validate_immutable(test_input):
                return False
            
            # Execute composition to verify no side effects
            with _composition_lock:
                result = self._composition(test_input)
                
            # Verify result immutability
            if not atomic_validate_immutable(result):
                return False
            
            self._validated = True
            return True
            
        except Exception:
            return False
    
    def __call__(self, *args, **kwargs):
        """Execute composition with thread safety"""
        if not self.validate_purity():
            raise ValidationError(
                "purity_violation",
                "Composition failed purity validation"
            )
        
        with _composition_lock:
            return self._composition(*args, **kwargs)

def linear_compose(*functions: Callable) -> LinearComposition:
    """
    Create linear-compliant function composition
    
    Args:
        *functions: Functions to compose (right-to-left evaluation)
        
    Returns:
        LinearComposition object with validated composition
        
    Thread Safety: Yes - creates isolated composition object
    """
    return LinearComposition(*functions)

def linear_pipe(*functions: Callable) -> LinearComposition:
    """
    Create linear-compliant function pipeline (left-to-right)
    
    Args:
        *functions: Functions to pipe (left-to-right evaluation)
        
    Returns:
        LinearComposition object with validated pipeline
        
    Thread Safety: Yes - creates isolated composition object
    """
    return LinearComposition(*reversed(functions))

def get_domain_exports():
    """Export composition operations for registration"""
    return {
        'LinearComposition': LinearComposition,
        'linear_compose': linear_compose,
        'linear_pipe': linear_pipe,
    }

# Validate composition module integrity
def validate_composition_module() -> bool:
    """Validate composition module maintains linear constraints"""
    try:
        # Test basic composition
        def add_one(x): return x + 1
        def multiply_two(x): return x * 2
        
        composed = linear_compose(multiply_two, add_one)
        if not composed.validate_purity():
            return False
        
        result = composed(3)
        if result != 8:  # (3 + 1) * 2 = 8
            return False
        
        # Test pipeline
        piped = linear_pipe(add_one, multiply_two)
        if not piped.validate_purity():
            return False
        
        if piped(3) != 8:  # (3 + 1) * 2 = 8
            return False
        
        return True
        
    except Exception:
        return False

# Self-validation on module load
if not validate_composition_module():
    raise RuntimeError("Composition module validation failed")
