#!/usr/bin/env python3
"""
pyics/core/registry/global/data_types.py
Pyics Core Domain Data Types: registry/global

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: registry/global
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for registry/global domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the registry/global
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class Registry/GlobalStatus(Enum):
    """Status enumeration for registry/global domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class Registry/GlobalPriority(Enum):
    """Priority levels for registry/global domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class Registry/GlobalEntity:
    """
    Base entity for registry/global domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: Registry/GlobalStatus = Registry/GlobalStatus.INITIALIZED
    priority: Registry/GlobalPriority = Registry/GlobalPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Registry/GlobalConfig:
    """
    Configuration data structure for registry/global domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Registry/GlobalResult:
    """
    Result container for registry/global domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class Registry/GlobalProcessor(Protocol):
    """Protocol for registry/global domain processors"""
    
    def process(self, entity: Registry/GlobalEntity) -> Registry/GlobalResult:
        """Process a registry/global entity"""
        ...
    
    def validate(self, entity: Registry/GlobalEntity) -> bool:
        """Validate a registry/global entity"""
        ...

class Registry/GlobalRepository(Protocol):
    """Protocol for registry/global domain data repositories"""
    
    def store(self, entity: Registry/GlobalEntity) -> bool:
        """Store a registry/global entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[Registry/GlobalEntity]:
        """Retrieve a registry/global entity by ID"""
        ...
    
    def list_all(self) -> List[Registry/GlobalEntity]:
        """List all registry/global entities"""
        ...

# Type aliases for complex structures
Registry/GlobalCollection = List[Registry/GlobalEntity]
Registry/GlobalIndex = Dict[str, Registry/GlobalEntity]
Registry/GlobalFilter = Dict[str, Any]

# Export interface
__all__ = [
    'Registry/GlobalStatus',
    'Registry/GlobalPriority',
    'Registry/GlobalEntity',
    'Registry/GlobalConfig',
    'Registry/GlobalResult',
    'Registry/GlobalProcessor',
    'Registry/GlobalRepository',
    'Registry/GlobalCollection',
    'Registry/GlobalIndex',
    'Registry/GlobalFilter',
]

# [EOF] - End of registry/global data_types.py module
