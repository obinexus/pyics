#!/usr/bin/env python3
"""
pyics/core/transforms/aggregation/data_types.py
Pyics Core Domain Data Types: transforms/aggregation

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: transforms/aggregation
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for transforms/aggregation domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the transforms/aggregation
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class Transforms/AggregationStatus(Enum):
    """Status enumeration for transforms/aggregation domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class Transforms/AggregationPriority(Enum):
    """Priority levels for transforms/aggregation domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class Transforms/AggregationEntity:
    """
    Base entity for transforms/aggregation domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: Transforms/AggregationStatus = Transforms/AggregationStatus.INITIALIZED
    priority: Transforms/AggregationPriority = Transforms/AggregationPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Transforms/AggregationConfig:
    """
    Configuration data structure for transforms/aggregation domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Transforms/AggregationResult:
    """
    Result container for transforms/aggregation domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class Transforms/AggregationProcessor(Protocol):
    """Protocol for transforms/aggregation domain processors"""
    
    def process(self, entity: Transforms/AggregationEntity) -> Transforms/AggregationResult:
        """Process a transforms/aggregation entity"""
        ...
    
    def validate(self, entity: Transforms/AggregationEntity) -> bool:
        """Validate a transforms/aggregation entity"""
        ...

class Transforms/AggregationRepository(Protocol):
    """Protocol for transforms/aggregation domain data repositories"""
    
    def store(self, entity: Transforms/AggregationEntity) -> bool:
        """Store a transforms/aggregation entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[Transforms/AggregationEntity]:
        """Retrieve a transforms/aggregation entity by ID"""
        ...
    
    def list_all(self) -> List[Transforms/AggregationEntity]:
        """List all transforms/aggregation entities"""
        ...

# Type aliases for complex structures
Transforms/AggregationCollection = List[Transforms/AggregationEntity]
Transforms/AggregationIndex = Dict[str, Transforms/AggregationEntity]
Transforms/AggregationFilter = Dict[str, Any]

# Export interface
__all__ = [
    'Transforms/AggregationStatus',
    'Transforms/AggregationPriority',
    'Transforms/AggregationEntity',
    'Transforms/AggregationConfig',
    'Transforms/AggregationResult',
    'Transforms/AggregationProcessor',
    'Transforms/AggregationRepository',
    'Transforms/AggregationCollection',
    'Transforms/AggregationIndex',
    'Transforms/AggregationFilter',
]

# [EOF] - End of transforms/aggregation data_types.py module
