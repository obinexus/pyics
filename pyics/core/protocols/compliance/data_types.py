#!/usr/bin/env python3
"""
pyics/core/protocols/data_types.py
Pyics Core Domain Data Types: protocols

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: protocols
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for protocols domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the protocols
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class ProtocolsStatus(Enum):
    """Status enumeration for protocols domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class ProtocolsPriority(Enum):
    """Priority levels for protocols domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class ProtocolsEntity:
    """
    Base entity for protocols domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: ProtocolsStatus = ProtocolsStatus.INITIALIZED
    priority: ProtocolsPriority = ProtocolsPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ProtocolsConfig:
    """
    Configuration data structure for protocols domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ProtocolsResult:
    """
    Result container for protocols domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class ProtocolsProcessor(Protocol):
    """Protocol for protocols domain processors"""
    
    def process(self, entity: ProtocolsEntity) -> ProtocolsResult:
        """Process a protocols entity"""
        ...
    
    def validate(self, entity: ProtocolsEntity) -> bool:
        """Validate a protocols entity"""
        ...

class ProtocolsRepository(Protocol):
    """Protocol for protocols domain data repositories"""
    
    def store(self, entity: ProtocolsEntity) -> bool:
        """Store a protocols entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[ProtocolsEntity]:
        """Retrieve a protocols entity by ID"""
        ...
    
    def list_all(self) -> List[ProtocolsEntity]:
        """List all protocols entities"""
        ...

# Type aliases for complex structures
ProtocolsCollection = List[ProtocolsEntity]
ProtocolsIndex = Dict[str, ProtocolsEntity]
ProtocolsFilter = Dict[str, Any]

# Export interface
__all__ = [
    'ProtocolsStatus',
    'ProtocolsPriority',
    'ProtocolsEntity',
    'ProtocolsConfig',
    'ProtocolsResult',
    'ProtocolsProcessor',
    'ProtocolsRepository',
    'ProtocolsCollection',
    'ProtocolsIndex',
    'ProtocolsFilter',
]

# [EOF] - End of protocols data_types.py module
