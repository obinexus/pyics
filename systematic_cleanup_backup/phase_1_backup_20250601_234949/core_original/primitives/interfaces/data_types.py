#!/usr/bin/env python3
"""
pyics/core/primitives/data_types.py
Pyics Core Domain Data Types: primitives

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: primitives
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for primitives domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the primitives
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class PrimitivesStatus(Enum):
    """Status enumeration for primitives domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class PrimitivesPriority(Enum):
    """Priority levels for primitives domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class PrimitivesEntity:
    """
    Base entity for primitives domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: PrimitivesStatus = PrimitivesStatus.INITIALIZED
    priority: PrimitivesPriority = PrimitivesPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PrimitivesConfig:
    """
    Configuration data structure for primitives domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PrimitivesResult:
    """
    Result container for primitives domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class PrimitivesProcessor(Protocol):
    """Protocol for primitives domain processors"""
    
    def process(self, entity: PrimitivesEntity) -> PrimitivesResult:
        """Process a primitives entity"""
        ...
    
    def validate(self, entity: PrimitivesEntity) -> bool:
        """Validate a primitives entity"""
        ...

class PrimitivesRepository(Protocol):
    """Protocol for primitives domain data repositories"""
    
    def store(self, entity: PrimitivesEntity) -> bool:
        """Store a primitives entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[PrimitivesEntity]:
        """Retrieve a primitives entity by ID"""
        ...
    
    def list_all(self) -> List[PrimitivesEntity]:
        """List all primitives entities"""
        ...

# Type aliases for complex structures
PrimitivesCollection = List[PrimitivesEntity]
PrimitivesIndex = Dict[str, PrimitivesEntity]
PrimitivesFilter = Dict[str, Any]

# Export interface
__all__ = [
    'PrimitivesStatus',
    'PrimitivesPriority',
    'PrimitivesEntity',
    'PrimitivesConfig',
    'PrimitivesResult',
    'PrimitivesProcessor',
    'PrimitivesRepository',
    'PrimitivesCollection',
    'PrimitivesIndex',
    'PrimitivesFilter',
]

# [EOF] - End of primitives data_types.py module
