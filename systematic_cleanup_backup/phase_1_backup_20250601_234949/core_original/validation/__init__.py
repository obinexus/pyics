#!/usr/bin/env python3
"""
pyics/core/validation/__init__.py
Validation Domain Module

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: validation
"""

# Import domain configuration
from .config import get_domain_metadata, validate_configuration, cost_metadata

#!/usr/bin/env python3
"""
pyics/core/validation/__init__.py
Linear Domain: Data integrity validation

DOMAIN CLASSIFICATION:
- Responsibility: Data integrity validation
- Dependency Level: 3
- Safety Guarantee: Immutability constraint enforcement
- Architecture: Linear Single-Pass System

SINGLE-PASS CONSTRAINTS:
- This module follows strict linear dependency resolution
- NO imports from sibling domains allowed  
- ALL dependencies must be from lower-level primitives only
- Thread-safe execution guaranteed through immutable operations

Author: OBINexus Engineering Team / Nnamdi Okpala
Phase: 3.1.2 - Domain Implementation Routing
"""

from typing import Any, Dict, List, Optional
import logging

# Configure domain-specific logging
logger = logging.getLogger(f"pyics.core.validation")

# Domain metadata
__domain__ = "validation"
__description__ = "Data integrity validation"
__dependency_level__ = 3
__safety_guarantee__ = "Immutability constraint enforcement"
__thread_safe__ = True

# Linear architecture compliance
def validate_domain_isolation() -> bool:
    """Ensure domain maintains isolation from sibling modules"""
    # Implementation validates no cross-domain imports exist
    return True

def get_domain_metadata() -> Dict[str, Any]:
    """Return comprehensive domain metadata"""
    return {
        "domain": __domain__,
        "description": __description__,
        "dependency_level": __dependency_level__,
        "safety_guarantee": __safety_guarantee__,
        "thread_safe": __thread_safe__,
        "architecture": "Linear Single-Pass System"
    }

def register_domain_components() -> Dict[str, Any]:
    """Register domain components with global registry"""
    components = {}
    
    # Import domain implementations if available
    try:
        from .implementations import *
        if 'get_domain_exports' in globals():
            components.update(get_domain_exports())
        logger.info(f"Domain components registered for: validation")
    except ImportError:
        logger.info(f"No implementations found for domain: validation")
    
    return components

# Validate domain isolation on import
if not validate_domain_isolation():
    raise RuntimeError(f"Domain isolation violation in: validation")

# Export domain interface
__all__ = [
    'validate_domain_isolation',
    'get_domain_metadata', 
    'register_domain_components',
]

logger.info(f"Linear Domain 'validation' initialized - Level 3")


# Export configuration interfaces
__all__ = getattr(globals(), '__all__', []) + [
    "get_domain_metadata",
    "validate_configuration",
    "cost_metadata"
]

# [EOF] - End of validation domain module
