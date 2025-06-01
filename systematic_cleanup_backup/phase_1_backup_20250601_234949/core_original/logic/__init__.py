#!/usr/bin/env python3
"""
pyics/core/logic/__init__.py
Logic Domain Module

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: logic
"""

# Import domain configuration
from .config import get_domain_metadata, validate_configuration, cost_metadata



# Export configuration interfaces
__all__ = getattr(globals(), '__all__', []) + [
    "get_domain_metadata",
    "validate_configuration",
    "cost_metadata"
]

# [EOF] - End of logic domain module
