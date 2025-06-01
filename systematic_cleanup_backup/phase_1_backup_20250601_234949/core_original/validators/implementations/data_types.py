#!/usr/bin/env python3
"""
pyics/core/validators/data_types.py
Pyics Core Domain Data Types: validators

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: validators
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for validators domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the validators
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class ValidatorsStatus(Enum):
    """Status enumeration for validators domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class ValidatorsPriority(Enum):
    """Priority levels for validators domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class ValidatorsEntity:
    """
    Base entity for validators domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: ValidatorsStatus = ValidatorsStatus.INITIALIZED
    priority: ValidatorsPriority = ValidatorsPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ValidatorsConfig:
    """
    Configuration data structure for validators domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ValidatorsResult:
    """
    Result container for validators domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class ValidatorsProcessor(Protocol):
    """Protocol for validators domain processors"""
    
    def process(self, entity: ValidatorsEntity) -> ValidatorsResult:
        """Process a validators entity"""
        ...
    
    def validate(self, entity: ValidatorsEntity) -> bool:
        """Validate a validators entity"""
        ...

class ValidatorsRepository(Protocol):
    """Protocol for validators domain data repositories"""
    
    def store(self, entity: ValidatorsEntity) -> bool:
        """Store a validators entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[ValidatorsEntity]:
        """Retrieve a validators entity by ID"""
        ...
    
    def list_all(self) -> List[ValidatorsEntity]:
        """List all validators entities"""
        ...

# Type aliases for complex structures
ValidatorsCollection = List[ValidatorsEntity]
ValidatorsIndex = Dict[str, ValidatorsEntity]
ValidatorsFilter = Dict[str, Any]

# Export interface
__all__ = [
    'ValidatorsStatus',
    'ValidatorsPriority',
    'ValidatorsEntity',
    'ValidatorsConfig',
    'ValidatorsResult',
    'ValidatorsProcessor',
    'ValidatorsRepository',
    'ValidatorsCollection',
    'ValidatorsIndex',
    'ValidatorsFilter',
]

# [EOF] - End of validators data_types.py module
