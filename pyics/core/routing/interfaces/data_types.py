#!/usr/bin/env python3
"""
pyics/core/routing/data_types.py
Pyics Core Domain Data Types: routing

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: routing
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for routing domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the routing
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class RoutingStatus(Enum):
    """Status enumeration for routing domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class RoutingPriority(Enum):
    """Priority levels for routing domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class RoutingEntity:
    """
    Base entity for routing domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: RoutingStatus = RoutingStatus.INITIALIZED
    priority: RoutingPriority = RoutingPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RoutingConfig:
    """
    Configuration data structure for routing domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RoutingResult:
    """
    Result container for routing domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class RoutingProcessor(Protocol):
    """Protocol for routing domain processors"""
    
    def process(self, entity: RoutingEntity) -> RoutingResult:
        """Process a routing entity"""
        ...
    
    def validate(self, entity: RoutingEntity) -> bool:
        """Validate a routing entity"""
        ...

class RoutingRepository(Protocol):
    """Protocol for routing domain data repositories"""
    
    def store(self, entity: RoutingEntity) -> bool:
        """Store a routing entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[RoutingEntity]:
        """Retrieve a routing entity by ID"""
        ...
    
    def list_all(self) -> List[RoutingEntity]:
        """List all routing entities"""
        ...

# Type aliases for complex structures
RoutingCollection = List[RoutingEntity]
RoutingIndex = Dict[str, RoutingEntity]
RoutingFilter = Dict[str, Any]

# Export interface
__all__ = [
    'RoutingStatus',
    'RoutingPriority',
    'RoutingEntity',
    'RoutingConfig',
    'RoutingResult',
    'RoutingProcessor',
    'RoutingRepository',
    'RoutingCollection',
    'RoutingIndex',
    'RoutingFilter',
]

# [EOF] - End of routing data_types.py module
