#!/usr/bin/env python3
"""
pyics/core/validators/implementations/data_integrity.py
Linear Validators - Data Integrity Checking (Dependency Level 3)

DEPENDENCIES: primitives, protocols, composition
"""

from typing import Any, Callable, TypeVar
from ...primitives.implementations.atomic_operations import atomic_validate_immutable
from ...protocols.implementations.linear_interfaces import ValidationError

T = TypeVar('T')

def validate_data_integrity(data: Any) -> bool:
    """Validate data maintains linear integrity constraints"""
    return atomic_validate_immutable(data)

def create_integrity_validator(constraint: Callable[[Any], bool]) -> Callable[[T], bool]:
    """Create integrity validator with linear compliance"""
    def validator(data: T) -> bool:
        if not validate_data_integrity(data):
            return False
        return constraint(data)
    return validator

def get_domain_exports():
    return {
        'validate_data_integrity': validate_data_integrity,
        'create_integrity_validator': create_integrity_validator,
    }
