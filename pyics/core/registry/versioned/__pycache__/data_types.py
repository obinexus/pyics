#!/usr/bin/env python3
"""
pyics/core/registry/versioned/data_types.py
Pyics Core Domain Data Types: registry/versioned

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: registry/versioned
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for registry/versioned domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the registry/versioned
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class Registry/VersionedStatus(Enum):
    """Status enumeration for registry/versioned domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class Registry/VersionedPriority(Enum):
    """Priority levels for registry/versioned domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class Registry/VersionedEntity:
    """
    Base entity for registry/versioned domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: Registry/VersionedStatus = Registry/VersionedStatus.INITIALIZED
    priority: Registry/VersionedPriority = Registry/VersionedPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Registry/VersionedConfig:
    """
    Configuration data structure for registry/versioned domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Registry/VersionedResult:
    """
    Result container for registry/versioned domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class Registry/VersionedProcessor(Protocol):
    """Protocol for registry/versioned domain processors"""
    
    def process(self, entity: Registry/VersionedEntity) -> Registry/VersionedResult:
        """Process a registry/versioned entity"""
        ...
    
    def validate(self, entity: Registry/VersionedEntity) -> bool:
        """Validate a registry/versioned entity"""
        ...

class Registry/VersionedRepository(Protocol):
    """Protocol for registry/versioned domain data repositories"""
    
    def store(self, entity: Registry/VersionedEntity) -> bool:
        """Store a registry/versioned entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[Registry/VersionedEntity]:
        """Retrieve a registry/versioned entity by ID"""
        ...
    
    def list_all(self) -> List[Registry/VersionedEntity]:
        """List all registry/versioned entities"""
        ...

# Type aliases for complex structures
Registry/VersionedCollection = List[Registry/VersionedEntity]
Registry/VersionedIndex = Dict[str, Registry/VersionedEntity]
Registry/VersionedFilter = Dict[str, Any]

# Export interface
__all__ = [
    'Registry/VersionedStatus',
    'Registry/VersionedPriority',
    'Registry/VersionedEntity',
    'Registry/VersionedConfig',
    'Registry/VersionedResult',
    'Registry/VersionedProcessor',
    'Registry/VersionedRepository',
    'Registry/VersionedCollection',
    'Registry/VersionedIndex',
    'Registry/VersionedFilter',
]

# [EOF] - End of registry/versioned data_types.py module
