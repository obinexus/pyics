#!/usr/bin/env python3
"""
pyics/core/logic/data_types.py
Pyics Core Domain Data Types: logic

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: logic
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for logic domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the logic
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class LogicStatus(Enum):
    """Status enumeration for logic domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class LogicPriority(Enum):
    """Priority levels for logic domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class LogicEntity:
    """
    Base entity for logic domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: LogicStatus = LogicStatus.INITIALIZED
    priority: LogicPriority = LogicPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class LogicConfig:
    """
    Configuration data structure for logic domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class LogicResult:
    """
    Result container for logic domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class LogicProcessor(Protocol):
    """Protocol for logic domain processors"""
    
    def process(self, entity: LogicEntity) -> LogicResult:
        """Process a logic entity"""
        ...
    
    def validate(self, entity: LogicEntity) -> bool:
        """Validate a logic entity"""
        ...

class LogicRepository(Protocol):
    """Protocol for logic domain data repositories"""
    
    def store(self, entity: LogicEntity) -> bool:
        """Store a logic entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[LogicEntity]:
        """Retrieve a logic entity by ID"""
        ...
    
    def list_all(self) -> List[LogicEntity]:
        """List all logic entities"""
        ...

# Type aliases for complex structures
LogicCollection = List[LogicEntity]
LogicIndex = Dict[str, LogicEntity]
LogicFilter = Dict[str, Any]

# Export interface
__all__ = [
    'LogicStatus',
    'LogicPriority',
    'LogicEntity',
    'LogicConfig',
    'LogicResult',
    'LogicProcessor',
    'LogicRepository',
    'LogicCollection',
    'LogicIndex',
    'LogicFilter',
]

# [EOF] - End of logic data_types.py module
