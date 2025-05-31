#!/usr/bin/env python3
"""
pyics/core/logic/composition/data_types.py
Pyics Core Domain Data Types: logic/composition

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: logic/composition
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for logic/composition domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the logic/composition
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class Logic/CompositionStatus(Enum):
    """Status enumeration for logic/composition domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class Logic/CompositionPriority(Enum):
    """Priority levels for logic/composition domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class Logic/CompositionEntity:
    """
    Base entity for logic/composition domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: Logic/CompositionStatus = Logic/CompositionStatus.INITIALIZED
    priority: Logic/CompositionPriority = Logic/CompositionPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Logic/CompositionConfig:
    """
    Configuration data structure for logic/composition domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Logic/CompositionResult:
    """
    Result container for logic/composition domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class Logic/CompositionProcessor(Protocol):
    """Protocol for logic/composition domain processors"""
    
    def process(self, entity: Logic/CompositionEntity) -> Logic/CompositionResult:
        """Process a logic/composition entity"""
        ...
    
    def validate(self, entity: Logic/CompositionEntity) -> bool:
        """Validate a logic/composition entity"""
        ...

class Logic/CompositionRepository(Protocol):
    """Protocol for logic/composition domain data repositories"""
    
    def store(self, entity: Logic/CompositionEntity) -> bool:
        """Store a logic/composition entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[Logic/CompositionEntity]:
        """Retrieve a logic/composition entity by ID"""
        ...
    
    def list_all(self) -> List[Logic/CompositionEntity]:
        """List all logic/composition entities"""
        ...

# Type aliases for complex structures
Logic/CompositionCollection = List[Logic/CompositionEntity]
Logic/CompositionIndex = Dict[str, Logic/CompositionEntity]
Logic/CompositionFilter = Dict[str, Any]

# Export interface
__all__ = [
    'Logic/CompositionStatus',
    'Logic/CompositionPriority',
    'Logic/CompositionEntity',
    'Logic/CompositionConfig',
    'Logic/CompositionResult',
    'Logic/CompositionProcessor',
    'Logic/CompositionRepository',
    'Logic/CompositionCollection',
    'Logic/CompositionIndex',
    'Logic/CompositionFilter',
]

# [EOF] - End of logic/composition data_types.py module
