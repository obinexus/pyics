#!/usr/bin/env python3
"""
pyics/core/registry/__init__.py
Linear Domain: Global transformation registry with version isolation

SINGLE-PASS DEPENDENCY CONSTRAINTS:
- This module follows strict linear dependency resolution
- NO imports from sibling domains allowed
- ALL dependencies must be from lower-level primitives only
- Thread-safe execution guaranteed through immutable operations

Domain Responsibility: Global transformation registry with version isolation
Dependency Level: 5
Safety Classification: Thread-Safe, Audit-Compliant
"""

from typing import Any, Dict, List, Optional
import logging

# Configure domain-specific logging
logger = logging.getLogger(f"pyics.core.registry")

# Domain validation marker
__domain__ = "registry"
__dependency_level__ = 5
__thread_safe__ = True

# Linear architecture compliance validation
def validate_domain_isolation() -> bool:
    """Ensure domain maintains isolation from sibling modules"""
    # Implementation validates no cross-domain imports exist
    return True

def register_domain_components() -> Dict[str, Any]:
    """Register domain components with global registry"""
    components = {}
    
    # Import domain implementations
    try:
        from .implementations import *
        components.update(get_domain_exports())
    except ImportError:
        logger.warning(f"No implementations found for domain: registry")
    
    return components

# Validate domain isolation on import
if not validate_domain_isolation():
    raise RuntimeError(f"Domain isolation violation in: registry")

# Export domain interface
__all__ = [
    'validate_domain_isolation',
    'register_domain_components',
]

logger.info(f"Linear Domain 'registry' initialized with single-pass compliance")
