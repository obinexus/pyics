#!/usr/bin/env python3
"""
pyics/core/transformations/data_types.py
Pyics Core Domain Data Types: transformations

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: transformations
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for transformations domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the transformations
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class TransformationsStatus(Enum):
    """Status enumeration for transformations domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class TransformationsPriority(Enum):
    """Priority levels for transformations domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class TransformationsEntity:
    """
    Base entity for transformations domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: TransformationsStatus = TransformationsStatus.INITIALIZED
    priority: TransformationsPriority = TransformationsPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TransformationsConfig:
    """
    Configuration data structure for transformations domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TransformationsResult:
    """
    Result container for transformations domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class TransformationsProcessor(Protocol):
    """Protocol for transformations domain processors"""
    
    def process(self, entity: TransformationsEntity) -> TransformationsResult:
        """Process a transformations entity"""
        ...
    
    def validate(self, entity: TransformationsEntity) -> bool:
        """Validate a transformations entity"""
        ...

class TransformationsRepository(Protocol):
    """Protocol for transformations domain data repositories"""
    
    def store(self, entity: TransformationsEntity) -> bool:
        """Store a transformations entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[TransformationsEntity]:
        """Retrieve a transformations entity by ID"""
        ...
    
    def list_all(self) -> List[TransformationsEntity]:
        """List all transformations entities"""
        ...

# Type aliases for complex structures
TransformationsCollection = List[TransformationsEntity]
TransformationsIndex = Dict[str, TransformationsEntity]
TransformationsFilter = Dict[str, Any]

# Export interface
__all__ = [
    'TransformationsStatus',
    'TransformationsPriority',
    'TransformationsEntity',
    'TransformationsConfig',
    'TransformationsResult',
    'TransformationsProcessor',
    'TransformationsRepository',
    'TransformationsCollection',
    'TransformationsIndex',
    'TransformationsFilter',
]

# [EOF] - End of transformations data_types.py module
