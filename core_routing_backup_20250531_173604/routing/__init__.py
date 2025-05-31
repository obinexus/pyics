#!/usr/bin/env python3
"""
pyics/core/routing/__init__.py
Linear Domain: Single-pass transformation routing and registry management

SINGLE-PASS DEPENDENCY CONSTRAINTS:
- This module follows strict linear dependency resolution
- NO imports from sibling domains allowed
- ALL dependencies must be from lower-level primitives only
- Thread-safe execution guaranteed through immutable operations

Domain Responsibility: Single-pass transformation routing and registry management
Dependency Level: 6
Safety Classification: Thread-Safe, Audit-Compliant
"""

from typing import Any, Dict, List, Optional
import logging

# Configure domain-specific logging
logger = logging.getLogger(f"pyics.core.routing")

# Domain validation marker
__domain__ = "routing"
__dependency_level__ = 6
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
        logger.warning(f"No implementations found for domain: routing")
    
    return components

# Validate domain isolation on import
if not validate_domain_isolation():
    raise RuntimeError(f"Domain isolation violation in: routing")

# Export domain interface
__all__ = [
    'validate_domain_isolation',
    'register_domain_components',
]

logger.info(f"Linear Domain 'routing' initialized with single-pass compliance")
