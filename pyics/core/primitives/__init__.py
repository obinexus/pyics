#!/usr/bin/env python3
"""
pyics/core/primitives/__init__.py
Primitives Domain Module

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: primitives
Phase: 3.1.6.2 - Domain Modularization

PROBLEM SOLVED: Atomic operations providing thread-safe, deterministic building blocks with zero dependencies
SEPARATION RATIONALE: Must remain isolated to preserve atomic guarantees and avoid cross-domain contamination
MERGE POTENTIAL: PRESERVE

Public interface for primitives domain following single-responsibility principles
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
DOMAIN_NAME = "primitives"
DOMAIN_SPECIFICATION = {
    "priority_index": 1,
    "compute_time_weight": 0.1,
    "exposure_type": "core_internal",
    "thread_safe": True,
    "load_order": 10
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
        logger = logging.getLogger(f"pyics.core.primitives")
        logger.debug(f"Domain primitives loaded and validated successfully")
    else:
        import logging
        logger = logging.getLogger(f"pyics.core.primitives")
        logger.warning(f"Domain primitives loaded with validation warnings")
except Exception as e:
    import logging
    logger = logging.getLogger(f"pyics.core.primitives")
    logger.error(f"Domain primitives validation failed on load: {e}")

# [EOF] - End of primitives domain module
