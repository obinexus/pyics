#!/usr/bin/env python3
"""
pyics/core/registry/migration/data_types.py
Pyics Core Domain Data Types: registry/migration

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: registry/migration
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for registry/migration domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the registry/migration
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class Registry/MigrationStatus(Enum):
    """Status enumeration for registry/migration domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class Registry/MigrationPriority(Enum):
    """Priority levels for registry/migration domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class Registry/MigrationEntity:
    """
    Base entity for registry/migration domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: Registry/MigrationStatus = Registry/MigrationStatus.INITIALIZED
    priority: Registry/MigrationPriority = Registry/MigrationPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Registry/MigrationConfig:
    """
    Configuration data structure for registry/migration domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Registry/MigrationResult:
    """
    Result container for registry/migration domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class Registry/MigrationProcessor(Protocol):
    """Protocol for registry/migration domain processors"""
    
    def process(self, entity: Registry/MigrationEntity) -> Registry/MigrationResult:
        """Process a registry/migration entity"""
        ...
    
    def validate(self, entity: Registry/MigrationEntity) -> bool:
        """Validate a registry/migration entity"""
        ...

class Registry/MigrationRepository(Protocol):
    """Protocol for registry/migration domain data repositories"""
    
    def store(self, entity: Registry/MigrationEntity) -> bool:
        """Store a registry/migration entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[Registry/MigrationEntity]:
        """Retrieve a registry/migration entity by ID"""
        ...
    
    def list_all(self) -> List[Registry/MigrationEntity]:
        """List all registry/migration entities"""
        ...

# Type aliases for complex structures
Registry/MigrationCollection = List[Registry/MigrationEntity]
Registry/MigrationIndex = Dict[str, Registry/MigrationEntity]
Registry/MigrationFilter = Dict[str, Any]

# Export interface
__all__ = [
    'Registry/MigrationStatus',
    'Registry/MigrationPriority',
    'Registry/MigrationEntity',
    'Registry/MigrationConfig',
    'Registry/MigrationResult',
    'Registry/MigrationProcessor',
    'Registry/MigrationRepository',
    'Registry/MigrationCollection',
    'Registry/MigrationIndex',
    'Registry/MigrationFilter',
]

# [EOF] - End of registry/migration data_types.py module
