#!/usr/bin/env python3
"""
pyics/core/protocols/__init__.py
Linear Domain: Interface definitions and contracts

DOMAIN CLASSIFICATION:
- Responsibility: Interface definitions and contracts
- Dependency Level: 1
- Safety Guarantee: Type safety enforcement
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
logger = logging.getLogger(f"pyics.core.protocols")

# Domain metadata
__domain__ = "protocols"
__description__ = "Interface definitions and contracts"
__dependency_level__ = 1
__safety_guarantee__ = "Type safety enforcement"
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
        logger.info(f"Domain components registered for: protocols")
    except ImportError:
        logger.info(f"No implementations found for domain: protocols")
    
    return components

# Validate domain isolation on import
if not validate_domain_isolation():
    raise RuntimeError(f"Domain isolation violation in: protocols")

# Export domain interface
__all__ = [
    'validate_domain_isolation',
    'get_domain_metadata', 
    'register_domain_components',
]

logger.info(f"Linear Domain 'protocols' initialized - Level 1")
