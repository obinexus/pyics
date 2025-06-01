#!/usr/bin/env python3
"""
pyics/core/protocols/__init__.py
Protocols Domain Module

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: protocols
Phase: 3.1.6.2 - Domain Modularization

PROBLEM SOLVED: Defines all type-safe interfaces for cross-domain communication
SEPARATION RATIONALE: Interface-only, no implementation logic allowed
MERGE POTENTIAL: PRESERVE

Public interface for protocols domain following single-responsibility principles
and maintaining architectural isolation for deterministic behavior.
"""

# Import domain configuration
from .config import (
    get_domain_metadata,
    validate_configuration,
    cost_metadata,
    get_behavior_policy,
    update_behavior_policy
)

# Import core domain components
try:
    from .data_types import *
except ImportError:
    pass  # data_types module may not exist yet

try:
    from .operations import *
except ImportError:
    pass  # operations module may not exist yet

try:
    from .relations import *
except ImportError:
    pass  # relations module may not exist yet

# Domain metadata for external access
DOMAIN_NAME = "protocols"
DOMAIN_SPECIFICATION = {
    "priority_index": 1,
    "compute_time_weight": 0.05,
    "exposure_type": "version_required",
    "thread_safe": True,
    "load_order": 20
}

# Export configuration interfaces
__all__ = [
    "DOMAIN_NAME",
    "DOMAIN_SPECIFICATION", 
    "get_domain_metadata",
    "validate_configuration",
    "cost_metadata",
    "get_behavior_policy",
    "update_behavior_policy"
]

# Auto-validate domain on module load
try:
    if validate_configuration():
        import logging
        logger = logging.getLogger(f"pyics.core.protocols")
        logger.debug(f"Domain protocols loaded and validated successfully")
    else:
        import logging
        logger = logging.getLogger(f"pyics.core.protocols")
        logger.warning(f"Domain protocols loaded with validation warnings")
except Exception as e:
    import logging
    logger = logging.getLogger(f"pyics.core.protocols")
    logger.error(f"Domain protocols validation failed on load: {e}")

# [EOF] - End of protocols domain module
