#!/usr/bin/env python3
"""
pyics/core/transforms/pipeline/__pycache__/__init__.py
Pyics Core Domain: transforms/pipeline

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: transforms/pipeline
Responsibility: Domain initialization and export interface
Compute Weight: Minimal (import-time only)

PROBLEM SOLVED: Centralized domain module exports and initialization
DEPENDENCIES: Domain-specific modules (data_types, relations, operations)
THREAD SAFETY: Yes - static exports only
DETERMINISTIC: Yes - immutable module structure

This module provides the public interface for the transforms/pipeline domain,
exposing core data types, relations, and operations following DOP principles.
"""

from typing import Any

# Domain metadata
__domain__ = "transforms/pipeline"
__version__ = "1.0.0"
__compute_weight__ = "minimal"

# Module exports will be populated during integration
__all__: list[str] = []

# [EOF] - End of transforms/pipeline domain __init__.py
