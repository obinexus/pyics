#!/usr/bin/env python3
"""
pyics/core/transformations/implementations/pure_transforms.py
Linear Transformations - Pure Transform Functions (Dependency Level 4)

DEPENDENCIES: primitives, protocols, composition, validators
"""

from typing import Callable, TypeVar, Any
from ...composition.implementations.lambda_calculus import linear_compose
from ...validators.implementations.data_integrity import validate_data_integrity

T = TypeVar('T')
U = TypeVar('U')

def create_linear_transform(transform_func: Callable[[T], U]) -> Callable[[T], U]:
    """Create linear-compliant transformation with validation"""
    def linear_validated_transform(data: T) -> U:
        if not validate_data_integrity(data):
            raise ValueError("Input data failed linear integrity validation")
        
        result = transform_func(data)
        
        if not validate_data_integrity(result):
            raise ValueError("Transform result failed linear integrity validation")
        
        return result
    
    return linear_validated_transform

def get_domain_exports():
    return {
        'create_linear_transform': create_linear_transform,
    }
