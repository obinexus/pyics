#!/usr/bin/env python3
"""
pyics/core/validation/data_types.py
Pyics Core Domain Data Types: validation

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: validation
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for validation domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the validation
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class ValidationStatus(Enum):
    """Status enumeration for validation domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class ValidationPriority(Enum):
    """Priority levels for validation domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class ValidationEntity:
    """
    Base entity for validation domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: ValidationStatus = ValidationStatus.INITIALIZED
    priority: ValidationPriority = ValidationPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ValidationConfig:
    """
    Configuration data structure for validation domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ValidationResult:
    """
    Result container for validation domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class ValidationProcessor(Protocol):
    """Protocol for validation domain processors"""
    
    def process(self, entity: ValidationEntity) -> ValidationResult:
        """Process a validation entity"""
        ...
    
    def validate(self, entity: ValidationEntity) -> bool:
        """Validate a validation entity"""
        ...

class ValidationRepository(Protocol):
    """Protocol for validation domain data repositories"""
    
    def store(self, entity: ValidationEntity) -> bool:
        """Store a validation entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[ValidationEntity]:
        """Retrieve a validation entity by ID"""
        ...
    
    def list_all(self) -> List[ValidationEntity]:
        """List all validation entities"""
        ...

# Type aliases for complex structures
ValidationCollection = List[ValidationEntity]
ValidationIndex = Dict[str, ValidationEntity]
ValidationFilter = Dict[str, Any]

# Export interface
__all__ = [
    'ValidationStatus',
    'ValidationPriority',
    'ValidationEntity',
    'ValidationConfig',
    'ValidationResult',
    'ValidationProcessor',
    'ValidationRepository',
    'ValidationCollection',
    'ValidationIndex',
    'ValidationFilter',
]

# [EOF] - End of validation data_types.py module
