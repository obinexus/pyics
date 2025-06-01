#!/usr/bin/env python3
"""
pyics/core/structures/__init__.py
Structures Domain Module

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: structures
Phase: 3.1.6.2 - Domain Modularization

PROBLEM SOLVED: Defines immutable data containers for all calendar operations
SEPARATION RATIONALE: Must remain pure (dataclasses only) to enforce data immutability and safety
MERGE POTENTIAL: PRESERVE

Public interface for structures domain following single-responsibility principles
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
DOMAIN_NAME = "structures"
DOMAIN_SPECIFICATION = {
    "priority_index": 2,
    "compute_time_weight": 0.2,
    "exposure_type": "version_required",
    "thread_safe": True,
    "load_order": 30
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
        logger = logging.getLogger(f"pyics.core.structures")
        logger.debug(f"Domain structures loaded and validated successfully")
    else:
        import logging
        logger = logging.getLogger(f"pyics.core.structures")
        logger.warning(f"Domain structures loaded with validation warnings")
except Exception as e:
    import logging
    logger = logging.getLogger(f"pyics.core.structures")
    logger.error(f"Domain structures validation failed on load: {e}")

# [EOF] - End of structures domain module
