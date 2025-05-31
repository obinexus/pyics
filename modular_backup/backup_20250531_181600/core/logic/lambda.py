#!/usr/bin/env python3
"""
pyics/core/logic/lambda.py
Lambda Calculus Foundation - Mathematical Function Composition

This module provides the theoretical foundation for all Pyics transformations.
Every version-specific operation MUST route through these composition primitives.

Author: OBINexus Engineering Team / Nnamdi Okpala
Phase: 3.1 - Core Foundation Implementation
"""

from typing import Any, Callable, TypeVar
from functools import reduce, partial, wraps
import inspect

T = TypeVar('T')
U = TypeVar('U')

class PurityViolationError(Exception):
    """Raised when function violates DOP purity constraints"""
    pass

class Lambda:
    """Mathematical foundation for pure function composition"""
    
    @staticmethod
    def identity(x: T) -> T:
        """Identity function: λx.x"""
        return x
    
    @staticmethod
    def compose(*functions: Callable) -> Callable:
        """Function composition: (f ∘ g)(x) = f(g(x))"""
        def _compose_two(f: Callable, g: Callable) -> Callable:
            @wraps(f)
            def composed(*args, **kwargs):
                return f(g(*args, **kwargs))
            return composed
        
        if not functions:
            return Lambda.identity
        return reduce(_compose_two, functions)
    
    @staticmethod
    def pipe(*functions: Callable) -> Callable:
        """Left-to-right composition: pipe(f, g, h)(x) == h(g(f(x)))"""
        return Lambda.compose(*reversed(functions))
    
    @staticmethod
    def curry(func: Callable) -> Callable:
        """Transform f(x, y, z) into f(x)(y)(z)"""
        sig = inspect.signature(func)
        param_count = len(sig.parameters)
        
        def curried(*args):
            if len(args) >= param_count:
                return func(*args[:param_count])
            return lambda *more_args: curried(*(args + more_args))
        
        return curried

# Global composition registry
_TRANSFORM_REGISTRY = {}

def register_transform(name: str, version: str = "core"):
    """Decorator for registering pure transformations"""
    def decorator(func: Callable) -> Callable:
        key = f"{version}::{name}"
        _TRANSFORM_REGISTRY[key] = func
        return func
    return decorator

def get_transform(name: str, version: str = "core") -> Callable:
    """Retrieve registered transformation"""
    key = f"{version}::{name}"
    return _TRANSFORM_REGISTRY.get(key)

# Export core primitives
__all__ = ['Lambda', 'register_transform', 'get_transform', 'PurityViolationError']
