#!/usr/bin/env python3
"""
pyics/core/routing/implementations/execution_router.py
Linear Routing - Execution Coordination (Dependency Level 6)

DEPENDENCIES: All lower-level domains including registry
"""

from typing import Any, Callable
from ...registry.implementations.global_registry import GLOBAL_REGISTRY

class ExecutionRouter:
    """Routes execution through registered linear components"""
    
    def route_transformation(self, transform_key: str, data: Any) -> Any:
        """Route data through registered transformation"""
        transform = GLOBAL_REGISTRY.get(transform_key)
        if transform is None:
            raise ValueError(f"Transform not found: {transform_key}")
        return transform(data)

def get_domain_exports():
    return {
        'ExecutionRouter': ExecutionRouter,
    }
