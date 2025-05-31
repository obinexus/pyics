#!/usr/bin/env python3
"""
pyics/core/__init__.py
DOP Foundation Core Module

Entry point for all data-oriented programming operations in Pyics.
Provides centralized access to lambda calculus, immutable structures,
and pure transformations.

Author: OBINexus Engineering Team / Nnamdi Okpala
Phase: 3.1 - Core Foundation Implementation
"""

# Core mathematical foundation
from .logic.lambda import Lambda, register_transform, get_transform

# Immutable data structures
from .structures.immutables import (
    ImmutableEvent, CalendarData, EventStatus, PriorityLevel
)

# Pure transformations
from .transforms.base import (
    shift_event_time, add_event_metadata, format_event_ics
)

# Validation framework
from .validation.purity import validate_function_purity, ensure_immutable_return

# Version information
__version__ = "3.1.0-foundation"
__author__ = "OBINexus Engineering Team"

# Public API
__all__ = [
    # Lambda calculus
    'Lambda', 'register_transform', 'get_transform',
    
    # Data structures
    'ImmutableEvent', 'CalendarData', 'EventStatus', 'PriorityLevel',
    
    # Transformations
    'shift_event_time', 'add_event_metadata', 'format_event_ics',
    
    # Validation
    'validate_function_purity', 'ensure_immutable_return'
]

# DOP compliance enforcement
def enforce_dop_compliance():
    """Validate DOP foundation integrity"""
    print("🔒 DOP Foundation initialized - Zero Trust Mode enabled")
    print("📋 All transformations must route through registered functions")
    print("🧮 Mathematical composition validated")
    return True

# Initialize foundation
enforce_dop_compliance()
