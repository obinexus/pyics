#!/usr/bin/env python3
"""
pyics/core/safety/implementations/thread_safety.py
RIFT Safety - Thread Safety Utilities (Dependency Level 7)

Cross-cutting safety concerns with minimal dependencies
"""

import threading
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')

def thread_safe_operation(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator ensuring thread-safe operation execution"""
    operation_lock = threading.RLock()
    
    @wraps(func)
    def thread_safe_wrapper(*args, **kwargs) -> T:
        with operation_lock:
            return func(*args, **kwargs)
    
    return thread_safe_wrapper

def get_domain_exports():
    return {
        'thread_safe_operation': thread_safe_operation,
    }
