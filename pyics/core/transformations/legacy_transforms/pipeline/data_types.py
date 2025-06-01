#!/usr/bin/env python3
"""
pyics/core/transforms/data_types.py
Pyics Core Domain Data Types: transforms

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: transforms
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for transforms domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the transforms
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class TransformsStatus(Enum):
    """Status enumeration for transforms domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class TransformsPriority(Enum):
    """Priority levels for transforms domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class TransformsEntity:
    """
    Base entity for transforms domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: TransformsStatus = TransformsStatus.INITIALIZED
    priority: TransformsPriority = TransformsPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TransformsConfig:
    """
    Configuration data structure for transforms domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TransformsResult:
    """
    Result container for transforms domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class TransformsProcessor(Protocol):
    """Protocol for transforms domain processors"""
    
    def process(self, entity: TransformsEntity) -> TransformsResult:
        """Process a transforms entity"""
        ...
    
    def validate(self, entity: TransformsEntity) -> bool:
        """Validate a transforms entity"""
        ...

class TransformsRepository(Protocol):
    """Protocol for transforms domain data repositories"""
    
    def store(self, entity: TransformsEntity) -> bool:
        """Store a transforms entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[TransformsEntity]:
        """Retrieve a transforms entity by ID"""
        ...
    
    def list_all(self) -> List[TransformsEntity]:
        """List all transforms entities"""
        ...

# Type aliases for complex structures
TransformsCollection = List[TransformsEntity]
TransformsIndex = Dict[str, TransformsEntity]
TransformsFilter = Dict[str, Any]

# Export interface
__all__ = [
    'TransformsStatus',
    'TransformsPriority',
    'TransformsEntity',
    'TransformsConfig',
    'TransformsResult',
    'TransformsProcessor',
    'TransformsRepository',
    'TransformsCollection',
    'TransformsIndex',
    'TransformsFilter',
]

# [EOF] - End of transforms data_types.py module
