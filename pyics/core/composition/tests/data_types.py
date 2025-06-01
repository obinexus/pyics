#!/usr/bin/env python3
"""
pyics/core/composition/data_types.py
Pyics Core Domain Data Types: composition

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: composition
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for composition domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the composition
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class CompositionStatus(Enum):
    """Status enumeration for composition domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class CompositionPriority(Enum):
    """Priority levels for composition domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class CompositionEntity:
    """
    Base entity for composition domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: CompositionStatus = CompositionStatus.INITIALIZED
    priority: CompositionPriority = CompositionPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CompositionConfig:
    """
    Configuration data structure for composition domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CompositionResult:
    """
    Result container for composition domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class CompositionProcessor(Protocol):
    """Protocol for composition domain processors"""
    
    def process(self, entity: CompositionEntity) -> CompositionResult:
        """Process a composition entity"""
        ...
    
    def validate(self, entity: CompositionEntity) -> bool:
        """Validate a composition entity"""
        ...

class CompositionRepository(Protocol):
    """Protocol for composition domain data repositories"""
    
    def store(self, entity: CompositionEntity) -> bool:
        """Store a composition entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[CompositionEntity]:
        """Retrieve a composition entity by ID"""
        ...
    
    def list_all(self) -> List[CompositionEntity]:
        """List all composition entities"""
        ...

# Type aliases for complex structures
CompositionCollection = List[CompositionEntity]
CompositionIndex = Dict[str, CompositionEntity]
CompositionFilter = Dict[str, Any]

# Export interface
__all__ = [
    'CompositionStatus',
    'CompositionPriority',
    'CompositionEntity',
    'CompositionConfig',
    'CompositionResult',
    'CompositionProcessor',
    'CompositionRepository',
    'CompositionCollection',
    'CompositionIndex',
    'CompositionFilter',
]

# [EOF] - End of composition data_types.py module
