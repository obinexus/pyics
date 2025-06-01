#!/usr/bin/env python3
"""
pyics/core/refactored/data_types.py
Pyics Core Domain Data Types: refactored

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: refactored
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for refactored domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the refactored
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class RefactoredStatus(Enum):
    """Status enumeration for refactored domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class RefactoredPriority(Enum):
    """Priority levels for refactored domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class RefactoredEntity:
    """
    Base entity for refactored domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: RefactoredStatus = RefactoredStatus.INITIALIZED
    priority: RefactoredPriority = RefactoredPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RefactoredConfig:
    """
    Configuration data structure for refactored domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RefactoredResult:
    """
    Result container for refactored domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class RefactoredProcessor(Protocol):
    """Protocol for refactored domain processors"""
    
    def process(self, entity: RefactoredEntity) -> RefactoredResult:
        """Process a refactored entity"""
        ...
    
    def validate(self, entity: RefactoredEntity) -> bool:
        """Validate a refactored entity"""
        ...

class RefactoredRepository(Protocol):
    """Protocol for refactored domain data repositories"""
    
    def store(self, entity: RefactoredEntity) -> bool:
        """Store a refactored entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[RefactoredEntity]:
        """Retrieve a refactored entity by ID"""
        ...
    
    def list_all(self) -> List[RefactoredEntity]:
        """List all refactored entities"""
        ...

# Type aliases for complex structures
RefactoredCollection = List[RefactoredEntity]
RefactoredIndex = Dict[str, RefactoredEntity]
RefactoredFilter = Dict[str, Any]

# Export interface
__all__ = [
    'RefactoredStatus',
    'RefactoredPriority',
    'RefactoredEntity',
    'RefactoredConfig',
    'RefactoredResult',
    'RefactoredProcessor',
    'RefactoredRepository',
    'RefactoredCollection',
    'RefactoredIndex',
    'RefactoredFilter',
]

# [EOF] - End of refactored data_types.py module
