#!/usr/bin/env python3
"""
pyics/core/structures/data_types.py
Pyics Core Domain Data Types: structures

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: structures
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for structures domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the structures
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class StructuresStatus(Enum):
    """Status enumeration for structures domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class StructuresPriority(Enum):
    """Priority levels for structures domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class StructuresEntity:
    """
    Base entity for structures domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: StructuresStatus = StructuresStatus.INITIALIZED
    priority: StructuresPriority = StructuresPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class StructuresConfig:
    """
    Configuration data structure for structures domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class StructuresResult:
    """
    Result container for structures domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class StructuresProcessor(Protocol):
    """Protocol for structures domain processors"""
    
    def process(self, entity: StructuresEntity) -> StructuresResult:
        """Process a structures entity"""
        ...
    
    def validate(self, entity: StructuresEntity) -> bool:
        """Validate a structures entity"""
        ...

class StructuresRepository(Protocol):
    """Protocol for structures domain data repositories"""
    
    def store(self, entity: StructuresEntity) -> bool:
        """Store a structures entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[StructuresEntity]:
        """Retrieve a structures entity by ID"""
        ...
    
    def list_all(self) -> List[StructuresEntity]:
        """List all structures entities"""
        ...

# Type aliases for complex structures
StructuresCollection = List[StructuresEntity]
StructuresIndex = Dict[str, StructuresEntity]
StructuresFilter = Dict[str, Any]

# Export interface
__all__ = [
    'StructuresStatus',
    'StructuresPriority',
    'StructuresEntity',
    'StructuresConfig',
    'StructuresResult',
    'StructuresProcessor',
    'StructuresRepository',
    'StructuresCollection',
    'StructuresIndex',
    'StructuresFilter',
]

# [EOF] - End of structures data_types.py module
