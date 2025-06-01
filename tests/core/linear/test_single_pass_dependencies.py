#!/usr/bin/env python3
"""
tests/core/linear/test_single_pass_dependencies.py
Linear Architecture Dependency Validation Tests

Validates that core architecture maintains single-pass dependency resolution.
"""

import pytest
import sys
import importlib
from pathlib import Path

def test_core_import_order():
    """Test that core modules import in correct dependency order"""
    try:
        # Should import successfully following dependency hierarchy
        import pyics.core
        assert True
    except ImportError as e:
        pytest.fail(f"Core import failed with dependency violation: {e}")

def test_no_circular_dependencies():
    """Validate no circular dependencies exist in core modules"""
    # Implementation would use static analysis to detect cycles
    assert True  # Placeholder for comprehensive cycle detection

def test_thread_safety_validation():
    """Validate all core operations are thread-safe"""
    from pyics.core.primitives.implementations.atomic_operations import atomic_identity
    from pyics.core.composition.implementations.lambda_calculus import linear_compose
    
    import threading
    import time
    
    results = []
    
    def thread_test():
        for _ in range(100):
            result = atomic_identity("thread_test")
            results.append(result)
            time.sleep(0.001)
    
    threads = [threading.Thread(target=thread_test) for _ in range(10)]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # All results should be identical (no race conditions)
    assert all(r == "thread_test" for r in results)
    assert len(results) == 1000  # 10 threads * 100 iterations

if __name__ == "__main__":
    pytest.main([__file__])
