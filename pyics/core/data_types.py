#!/usr/bin/env python3
"""
pyics/core/pyics/data_types.py
Pyics Core Domain Data Types: pyics

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: pyics
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for pyics domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the pyics
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class PyicsStatus(Enum):
    """Status enumeration for pyics domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class PyicsPriority(Enum):
    """Priority levels for pyics domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class PyicsEntity:
    """
    Base entity for pyics domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: PyicsStatus = PyicsStatus.INITIALIZED
    priority: PyicsPriority = PyicsPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PyicsConfig:
    """
    Configuration data structure for pyics domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PyicsResult:
    """
    Result container for pyics domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class PyicsProcessor(Protocol):
    """Protocol for pyics domain processors"""
    
    def process(self, entity: PyicsEntity) -> PyicsResult:
        """Process a pyics entity"""
        ...
    
    def validate(self, entity: PyicsEntity) -> bool:
        """Validate a pyics entity"""
        ...

class PyicsRepository(Protocol):
    """Protocol for pyics domain data repositories"""
    
    def store(self, entity: PyicsEntity) -> bool:
        """Store a pyics entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[PyicsEntity]:
        """Retrieve a pyics entity by ID"""
        ...
    
    def list_all(self) -> List[PyicsEntity]:
        """List all pyics entities"""
        ...

# Type aliases for complex structures
PyicsCollection = List[PyicsEntity]
PyicsIndex = Dict[str, PyicsEntity]
PyicsFilter = Dict[str, Any]

# Export interface
__all__ = [
    'PyicsStatus',
    'PyicsPriority',
    'PyicsEntity',
    'PyicsConfig',
    'PyicsResult',
    'PyicsProcessor',
    'PyicsRepository',
    'PyicsCollection',
    'PyicsIndex',
    'PyicsFilter',
]

# [EOF] - End of pyics data_types.py module
