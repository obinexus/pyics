#!/usr/bin/env python3
"""
pyics/core/safety/data_types.py
Pyics Core Domain Data Types: safety

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: safety
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for safety domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the safety
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class SafetyStatus(Enum):
    """Status enumeration for safety domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class SafetyPriority(Enum):
    """Priority levels for safety domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class SafetyEntity:
    """
    Base entity for safety domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: SafetyStatus = SafetyStatus.INITIALIZED
    priority: SafetyPriority = SafetyPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SafetyConfig:
    """
    Configuration data structure for safety domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SafetyResult:
    """
    Result container for safety domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class SafetyProcessor(Protocol):
    """Protocol for safety domain processors"""
    
    def process(self, entity: SafetyEntity) -> SafetyResult:
        """Process a safety entity"""
        ...
    
    def validate(self, entity: SafetyEntity) -> bool:
        """Validate a safety entity"""
        ...

class SafetyRepository(Protocol):
    """Protocol for safety domain data repositories"""
    
    def store(self, entity: SafetyEntity) -> bool:
        """Store a safety entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[SafetyEntity]:
        """Retrieve a safety entity by ID"""
        ...
    
    def list_all(self) -> List[SafetyEntity]:
        """List all safety entities"""
        ...

# Type aliases for complex structures
SafetyCollection = List[SafetyEntity]
SafetyIndex = Dict[str, SafetyEntity]
SafetyFilter = Dict[str, Any]

# Export interface
__all__ = [
    'SafetyStatus',
    'SafetyPriority',
    'SafetyEntity',
    'SafetyConfig',
    'SafetyResult',
    'SafetyProcessor',
    'SafetyRepository',
    'SafetyCollection',
    'SafetyIndex',
    'SafetyFilter',
]

# [EOF] - End of safety data_types.py module
