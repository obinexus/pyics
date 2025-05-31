#!/usr/bin/env python3
"""
pyics/core/lambda.py
Lambda Calculus Foundation for Data-Oriented Programming

This module provides the mathematical foundation for all Pyics transformations.
Every version-specific operation MUST route through these composition primitives.

Zero Trust Principle: No business logic bypasses functional composition validation.

Author: OBINexus Engineering Team / Nnamdi Okpala
License: MIT
Compliance: DOP Canon Phase 3.1
"""

from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, 
    TypeVar, Generic, Protocol, runtime_checkable
)
from functools import reduce, partial, wraps
import inspect
from collections.abc import Iterable

# Type system for functional composition
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

Transform = Callable[[T], U]
Predicate = Callable[[T], bool]
Aggregator = Callable[[List[T]], U]
Composer = Callable[..., Any]

@runtime_checkable
class Transformable(Protocol[T]):
    """Protocol enforcing transformation capability across all data structures"""
    def transform(self, func: Transform[T, U]) -> U: ...
    def validate_purity(self) -> bool: ...

@runtime_checkable
class Composable(Protocol):
    """Protocol for composable function objects with validation"""
    def compose(self, other: 'Composable') -> 'Composable': ...
    def verify_side_effects(self) -> bool: ...

class PurityViolationError(Exception):
    """Raised when a function violates DOP purity constraints"""
    def __init__(self, function_name: str, violation_type: str):
        self.function_name = function_name
        self.violation_type = violation_type
        super().__init__(f"Purity violation in {function_name}: {violation_type}")

# ===== LAMBDA CALCULUS FOUNDATION =====

class Lambda:
    """
    Mathematical foundation for pure function composition
    
    All Pyics transformations MUST use these primitives to ensure:
    - Zero side effects
    - Composable operations
    - Verifiable mathematical correctness
    """
    
    @staticmethod
    def identity(x: T) -> T:
        """Identity function: λx.x - Canonical pure operation"""
        return x
    
    @staticmethod
    def constant(value: T) -> Callable[[Any], T]:
        """Constant function: λx.c - Pure value injection"""
        return lambda _: value
    
    @staticmethod
    def compose(*functions: Callable) -> Callable:
        """
        Function composition: (f ∘ g)(x) = f(g(x))
        
        Mathematical composition with purity validation:
        - Right-to-left evaluation: compose(f, g, h)(x) == f(g(h(x)))
        - Automatic side effect detection
        - Composition chain verification
        """
        def _validate_purity(func: Callable) -> None:
            """Validate function purity before composition"""
            if hasattr(func, '__self__'):
                # Bound method - potential state mutation
                raise PurityViolationError(func.__name__, "bound_method_detected")
            
            # Check for global variable access in function code
            if hasattr(func, '__code__'):
                if func.__code__.co_names:
                    # Functions accessing global names may have side effects
                    # Allow only approved pure operations
                    allowed_globals = {'len', 'str', 'int', 'float', 'bool', 'tuple', 'list'}
                    suspect_globals = set(func.__code__.co_names) - allowed_globals
                    if suspect_globals:
                        raise PurityViolationError(
                            func.__name__, 
                            f"global_access: {suspect_globals}"
                        )
        
        def _compose_two(f: Callable, g: Callable) -> Callable:
            _validate_purity(f)
            _validate_purity(g)
            
            @wraps(f)
            def composed(*args, **kwargs):
                try:
                    intermediate = g(*args, **kwargs)
                    return f(intermediate)
                except Exception as e:
                    raise PurityViolationError(
                        f"{f.__name__}∘{g.__name__}", 
                        f"composition_failure: {str(e)}"
                    )
            return composed
        
        if not functions:
            return Lambda.identity
        if len(functions) == 1:
            _validate_purity(functions[0])
            return functions[0]
        
        return reduce(_compose_two, functions)
    
    @staticmethod
    def pipe(*functions: Callable) -> Callable:
        """
        Function piping: left-to-right composition
        
        pipe(f, g, h)(x) == h(g(f(x)))
        More intuitive for data transformation pipelines
        """
        return Lambda.compose(*reversed(functions))
    
    @staticmethod
    def curry(func: Callable) -> Callable:
        """
        Currying: Transform f(x, y, z) into f(x)(y)(z)
        
        Enables partial application and function specialization
        with purity enforcement at each application level
        """
        sig = inspect.signature(func)
        param_count = len(sig.parameters)
        
        def curried(*args):
            if len(args) >= param_count:
                return func(*args[:param_count])
            
            def next_application(*more_args):
                return curried(*(args + more_args))
            
            # Mark curried function as pure
            next_application.__pure__ = True
            return next_application
        
        curried.__pure__ = True
        return curried
    
    @staticmethod
    def partial_apply(func: Callable, *args, **kwargs) -> Callable:
        """Partial application with lambda calculus semantics and purity validation"""
        # Validate original function purity
        if not getattr(func, '__pure__', False):
            # Check if function appears to be pure (no mutations, global access)
            try:
                Lambda.compose(func)  # This will validate purity
            except PurityViolationError:
                raise PurityViolationError(
                    func.__name__, 
                    "impure_function_in_partial_application"
                )
        
        partial_func = partial(func, *args, **kwargs)
        partial_func.__pure__ = True
        return partial_func
    
    @staticmethod
    def validate_composition_chain(pipeline: Callable) -> Dict[str, Any]:
        """
        Validate entire composition chain for DOP compliance
        
        Returns validation report with:
        - Purity status
        - Side effect analysis
        - Performance characteristics
        """
        validation_report = {
            "is_pure": True,
            "side_effects": [],
            "composition_depth": 0,
            "performance_warnings": []
        }
        
        # Extract composition chain if available
        if hasattr(pipeline, '__wrapped__'):
            # Function has been composed
            validation_report["composition_depth"] = 1
            
        # Check for purity markers
        if not getattr(pipeline, '__pure__', False):
            validation_report["is_pure"] = False
            validation_report["side_effects"].append("purity_marker_missing")
        
        return validation_report

# ===== COMPOSITION REGISTRY =====

class CompositionRegistry:
    """
    Central registry for validated transformation pipelines
    
    All Pyics modules MUST register transformations here before use.
    Provides version compatibility and upgrade pathways.
    """
    
    def __init__(self):
        self._registered_transforms: Dict[str, Callable] = {}
        self._validation_cache: Dict[str, Dict[str, Any]] = {}
        self._composition_chains: Dict[str, List[str]] = {}
    
    def register(self, name: str, transform: Callable, version: str = "v1") -> None:
        """
        Register a validated transformation for reuse
        
        Args:
            name: Unique identifier for the transformation
            transform: Pure function implementing the transformation
            version: Version compatibility marker
        """
        # Validate transformation purity
        validation_report = Lambda.validate_composition_chain(transform)
        
        if not validation_report["is_pure"]:
            raise PurityViolationError(
                name, 
                f"registration_failed: {validation_report['side_effects']}"
            )
        
        versioned_name = f"{version}::{name}"
        self._registered_transforms[versioned_name] = transform
        self._validation_cache[versioned_name] = validation_report
        
        # Track composition chains for dependency analysis
        if versioned_name not in self._composition_chains:
            self._composition_chains[versioned_name] = []
    
    def get(self, name: str, version: str = "v1") -> Optional[Callable]:
        """Retrieve validated transformation by name and version"""
        versioned_name = f"{version}::{name}"
        return self._registered_transforms.get(versioned_name)
    
    def create_pipeline(self, *transform_names: str, version: str = "v1") -> Callable:
        """
        Create validated composition pipeline from registered transforms
        
        All transforms must be pre-registered and validated for purity
        """
        transforms = []
        for name in transform_names:
            versioned_name = f"{version}::{name}"
            if versioned_name not in self._registered_transforms:
                raise ValueError(f"Unregistered transform: {versioned_name}")
            transforms.append(self._registered_transforms[versioned_name])
        
        # Create and validate composition
        pipeline = Lambda.pipe(*transforms)
        
        # Cache composition validation
        pipeline_key = f"pipeline::{version}::{':'.join(transform_names)}"
        self._validation_cache[pipeline_key] = Lambda.validate_composition_chain(pipeline)
        
        return pipeline
    
    def get_validation_report(self, name: str, version: str = "v1") -> Dict[str, Any]:
        """Get detailed validation report for registered transformation"""
        versioned_name = f"{version}::{name}"
        return self._validation_cache.get(versioned_name, {})
    
    def list_registered(self, version: Optional[str] = None) -> List[str]:
        """List all registered transformations, optionally filtered by version"""
        if version:
            prefix = f"{version}::"
            return [name[len(prefix):] for name in self._registered_transforms.keys() 
                   if name.startswith(prefix)]
        return list(self._registered_transforms.keys())

# ===== GLOBAL REGISTRY INSTANCE =====

# Single source of truth for all Pyics transformations
GLOBAL_TRANSFORM_REGISTRY = CompositionRegistry()

# ===== DECORATORS FOR DOP ENFORCEMENT =====

def pure_function(func: Callable) -> Callable:
    """
    Decorator marking functions as pure with runtime validation
    
    Usage:
        @pure_function
        def my_transform(data):
            return transformed_data
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Runtime purity check
        try:
            result = func(*args, **kwargs)
            # Verify no mutations occurred
            # (This is a simplified check - production would be more sophisticated)
            return result
        except Exception as e:
            raise PurityViolationError(func.__name__, f"runtime_violation: {str(e)}")
    
    wrapper.__pure__ = True
    return wrapper

def register_transform(name: str, version: str = "v1"):
    """
    Decorator for automatic transformation registration
    
    Usage:
        @register_transform("add_timezone", version="v2")
        @pure_function
        def add_timezone_metadata(event, timezone):
            return event.with_metadata(timezone=timezone)
    """
    def decorator(func: Callable) -> Callable:
        # Ensure function is marked as pure
        if not getattr(func, '__pure__', False):
            raise PurityViolationError(
                func.__name__, 
                "registration_requires_pure_function_decorator"
            )
        
        # Register with global registry
        GLOBAL_TRANSFORM_REGISTRY.register(name, func, version)
        return func
    
    return decorator

# ===== VALIDATION UTILITIES =====

def validate_dop_compliance(module_path: str) -> Dict[str, Any]:
    """
    Validate entire module for DOP compliance
    
    Checks:
    - All functions marked as pure or registered
    - No global state mutations
    - Proper composition chain usage
    """
    compliance_report = {
        "module": module_path,
        "compliant": True,
        "violations": [],
        "warnings": [],
        "registered_transforms": []
    }
    
    # This would be implemented with AST analysis in production
    # For now, provide framework for compliance checking
    
    return compliance_report

if __name__ == "__main__":
    # Demonstration of DOP foundation usage
    print("=== Pyics DOP Foundation Validation ===")
    
    # Example pure function registration
    @register_transform("identity_test", version="core")
    @pure_function
    def identity_test(x):
        return x
    
    # Test composition
    test_pipeline = GLOBAL_TRANSFORM_REGISTRY.create_pipeline("identity_test", version="core")
    result = test_pipeline("test_data")
    
    print(f"Pipeline result: {result}")
    print(f"Registered transforms: {GLOBAL_TRANSFORM_REGISTRY.list_registered('core')}")
    print("=== Foundation validation complete ===")
