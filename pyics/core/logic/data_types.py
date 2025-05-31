#!/usr/bin/env python3
"""
pyics/core/core/data_types.py
Pyics Core Domain Data Types: core

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: core
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for core domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the core
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class CoreStatus(Enum):
    """Status enumeration for core domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class CorePriority(Enum):
    """Priority levels for core domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class CoreEntity:
    """
    Base entity for core domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: CoreStatus = CoreStatus.INITIALIZED
    priority: CorePriority = CorePriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CoreConfig:
    """
    Configuration data structure for core domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CoreResult:
    """
    Result container for core domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class CoreProcessor(Protocol):
    """Protocol for core domain processors"""
    
    def process(self, entity: CoreEntity) -> CoreResult:
        """Process a core entity"""
        ...
    
    def validate(self, entity: CoreEntity) -> bool:
        """Validate a core entity"""
        ...

class CoreRepository(Protocol):
    """Protocol for core domain data repositories"""
    
    def store(self, entity: CoreEntity) -> bool:
        """Store a core entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[CoreEntity]:
        """Retrieve a core entity by ID"""
        ...
    
    def list_all(self) -> List[CoreEntity]:
        """List all core entities"""
        ...

# Type aliases for complex structures
CoreCollection = List[CoreEntity]
CoreIndex = Dict[str, CoreEntity]
CoreFilter = Dict[str, Any]

# Export interface
__all__ = [
    'CoreStatus',
    'CorePriority',
    'CoreEntity',
    'CoreConfig',
    'CoreResult',
    'CoreProcessor',
    'CoreRepository',
    'CoreCollection',
    'CoreIndex',
    'CoreFilter',
]

# [EOF] - End of core data_types.py module
