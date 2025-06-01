#!/usr/bin/env python3
"""
pyics/core/transforms/__init__.py
Transforms Domain Module

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: transforms
"""

# Import domain configuration
from .config import get_domain_metadata, validate_configuration, cost_metadata



# Export configuration interfaces
__all__ = getattr(globals(), '__all__', []) + [
    "get_domain_metadata",
    "validate_configuration",
    "cost_metadata"
]

# [EOF] - End of transforms domain module
