#!/usr/bin/env python3
"""
pyics/core/protocols/implementations/linear_interfaces.py
Linear Protocols - Interface Definitions (Dependency Level 1)

Defines all interface contracts for linear single-pass architecture.
These protocols ensure type safety across composition chains.

DEPENDENCY CONSTRAINTS:
- May only import from primitives domain
- NO implementation logic allowed - interfaces only
- ALL protocols must support thread-safe operations
"""

from typing import Any, Protocol, TypeVar, Callable, runtime_checkable
from abc import ABC, abstractmethod

T = TypeVar('T')
U = TypeVar('U')

@runtime_checkable
class Transformable(Protocol[T]):
    """Protocol for objects supporting linear transformation chains"""
    
    def transform(self, func: Callable[[T], U]) -> U:
        """Apply transformation function maintaining immutability"""
        ...
    
    def validate_integrity(self) -> bool:
        """Validate object maintains required invariants"""
        ...

@runtime_checkable
class Composable(Protocol):
    """Protocol for composable function objects in single-pass chains"""
    
    def compose_with(self, other: 'Composable') -> 'Composable':
        """Compose with another function maintaining single-pass property"""
        ...
    
    def validate_purity(self) -> bool:
        """Validate function maintains purity constraints"""
        ...

@runtime_checkable
class Registrable(Protocol):
    """Protocol for objects registrable with global registry"""
    
    def get_registration_key(self) -> str:
        """Get unique registration identifier"""
        ...
    
    def get_dependency_level(self) -> int:
        """Get dependency level for ordering"""
        ...
    
    def validate_dependencies(self) -> bool:
        """Validate dependencies follow single-pass model"""
        ...

class ValidationError(Exception):
    """Raised when architecture constraints are violated"""
    
    def __init__(self, violation_type: str, details: str):
        self.violation_type = violation_type
        self.details = details
        super().__init__(f"Validation Error ({violation_type}): {details}")

# Abstract base classes for domain implementations
class DomainBase(ABC):
    """Base class for all linear domain implementations"""
    
    @abstractmethod
    def get_domain_name(self) -> str:
        """Return domain name for identification"""
        pass
    
    @abstractmethod
    def get_dependency_level(self) -> int:
        """Return dependency level for ordering"""
        pass
    
    @abstractmethod
    def validate_single_pass(self) -> bool:
        """Validate domain follows single-pass constraints"""
        pass

def get_domain_exports():
    """Export protocol definitions for registration"""
    return {
        'Transformable': Transformable,
        'Composable': Composable,
        'Registrable': Registrable,
        'ValidationError': ValidationError,
        'DomainBase': DomainBase,
    }
