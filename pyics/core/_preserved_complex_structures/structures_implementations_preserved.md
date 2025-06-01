# Preserved Complex Structure: structures/implementations

Preservation timestamp: 2025-06-01T23:50:03.887266
Systematic cleanup phase: structure_flattening

## data_types.py
```python
#!/usr/bin/env python3
"""
pyics/core/structures/data_types.py
Pyics Core Domain Data Types: structures

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: structures
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for structures domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the structures
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class StructuresStatus(Enum):
    """Status enumeration for structures domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class StructuresPriority(Enum):
    """Priority levels for structures domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class StructuresEntity:
    """
    Base entity for structures domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: StructuresStatus = StructuresStatus.INITIALIZED
    priority: StructuresPriority = StructuresPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class StructuresConfig:
    """
    Configuration data structure for structures domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class StructuresResult:
    """
    Result container for structures domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class StructuresProcessor(Protocol):
    """Protocol for structures domain processors"""
    
    def process(self, entity: StructuresEntity) -> StructuresResult:
        """Process a structures entity"""
        ...
    
    def validate(self, entity: StructuresEntity) -> bool:
        """Validate a structures entity"""
        ...

class StructuresRepository(Protocol):
    """Protocol for structures domain data repositories"""
    
    def store(self, entity: StructuresEntity) -> bool:
        """Store a structures entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[StructuresEntity]:
        """Retrieve a structures entity by ID"""
        ...
    
    def list_all(self) -> List[StructuresEntity]:
        """List all structures entities"""
        ...

# Type aliases for complex structures
StructuresCollection = List[StructuresEntity]
StructuresIndex = Dict[str, StructuresEntity]
StructuresFilter = Dict[str, Any]

# Export interface
__all__ = [
    'StructuresStatus',
    'StructuresPriority',
    'StructuresEntity',
    'StructuresConfig',
    'StructuresResult',
    'StructuresProcessor',
    'StructuresRepository',
    'StructuresCollection',
    'StructuresIndex',
    'StructuresFilter',
]

# [EOF] - End of structures data_types.py module

```

## immutable_structures.py
```python
#!/usr/bin/env python3
"""
pyics/core/structures.py
Immutable Data Structures for Data-Oriented Programming

All Pyics data flows through these immutable structures. Version-specific
modules MUST use these canonical representations for state management.

Zero Trust Principle: No direct state mutation allowed outside transformation chains.

Author: OBINexus Engineering Team / Nnamdi Okpala
License: MIT
Compliance: DOP Canon Phase 3.1
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, FrozenSet
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from abc import ABC, abstractmethod

from .lambda import Transform, Transformable, T, U

# ===== CORE ENUMERATIONS =====

class EventStatus(Enum):
    """Immutable event status enumeration"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class PriorityLevel(Enum):
    """Calendar event priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class DistributionChannel(Enum):
    """Available distribution channels for calendar events"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    FILE_SYSTEM = "file_system"
    REST_API = "rest_api"

class ComplianceLevel(Enum):
    """Compliance requirement levels for audit tracking"""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    ENTERPRISE = "enterprise"
    REGULATORY = "regulatory"

# ===== VALIDATION PROTOCOLS =====

class ImmutableValidationError(Exception):
    """Raised when immutable data structure validation fails"""
    def __init__(self, structure_type: str, field: str, violation: str):
        self.structure_type = structure_type
        self.field = field
        self.violation = violation
        super().__init__(f"Validation failed in {structure_type}.{field}: {violation}")

def validate_immutable_field(value: Any, field_name: str, structure_type: str) -> None:
    """Validate field constraints for immutable structures"""
    if value is None:
        return  # None values handled by Optional typing
    
    # Check for mutable containers that could break immutability
    if isinstance(value, (list, dict, set)):
        raise ImmutableValidationError(
            structure_type, 
            field_name, 
            f"mutable_container_detected: {type(value).__name__}"
        )

# ===== CORE IMMUTABLE EVENT STRUCTURE =====

@dataclass(frozen=True)
class ImmutableEvent(Transformable):
    """
    Core immutable event structure following DOP principles
    
    All transformations return new instances, ensuring zero side effects.
    This is the canonical event representation across all Pyics versions.
    """
    uid: str
    summary: str
    start_time: datetime
    duration: timedelta
    description: str = ""
    status: EventStatus = EventStatus.DRAFT
    priority: PriorityLevel = PriorityLevel.MEDIUM
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    tags: FrozenSet[str] = field(default_factory=frozenset)
    compliance_level: ComplianceLevel = ComplianceLevel.BASIC
    
    def __post_init__(self):
        """Post-initialization validation for immutable constraints"""
        # Validate all fields for immutability compliance
        for field_name, field_value in self.__dict__.items():
            validate_immutable_field(field_value, field_name, "ImmutableEvent")
        
        # Validate business logic constraints
        if self.duration.total_seconds() <= 0:
            raise ImmutableValidationError(
                "ImmutableEvent", 
                "duration", 
                "duration_must_be_positive"
            )
        
        if not self.uid.strip():
            raise ImmutableValidationError(
                "ImmutableEvent", 
                "uid", 
                "uid_cannot_be_empty"
            )
    
    def transform(self, func: Transform['ImmutableEvent', T]) -> T:
        """Apply transformation function maintaining immutability"""
        return func(self)
    
    def validate_purity(self) -> bool:
        """Validate that this instance maintains immutable constraints"""
        try:
            # Attempt to access all fields to ensure they're still immutable
            for field_name in self.__dataclass_fields__:
                field_value = getattr(self, field_name)
                validate_immutable_field(field_value, field_name, "ImmutableEvent")
            return True
        except ImmutableValidationError:
            return False
    
    def with_summary(self, summary: str) -> 'ImmutableEvent':
        """Pure transformation: update summary"""
        return ImmutableEvent(
            uid=self.uid,
            summary=summary,
            start_time=self.start_time,
            duration=self.duration,
            description=self.description,
            status=self.status,
            priority=self.priority,
            metadata=self.metadata,
            tags=self.tags,
            compliance_level=self.compliance_level
        )
    
    def with_metadata(self, **new_metadata) -> 'ImmutableEvent':
        """Pure transformation: merge metadata"""
        # Convert existing metadata to dict for merging
        existing_meta = dict(self.metadata)
        merged_metadata = {**existing_meta, **new_metadata}
        
        # Convert back to immutable tuple representation
        metadata_tuple = tuple(sorted(merged_metadata.items()))
        
        return ImmutableEvent(
            uid=self.uid,
            summary=self.summary,
            start_time=self.start_time,
            duration=self.duration,
            description=self.description,
            status=self.status,
            priority=self.priority,
            metadata=metadata_tuple,
            tags=self.tags,
            compliance_level=self.compliance_level
        )
    
    def with_tags(self, *new_tags: str) -> 'ImmutableEvent':
        """Pure transformation: add tags"""
        combined_tags = self.tags | frozenset(new_tags)
        
        return ImmutableEvent(
            uid=self.uid,
            summary=self.summary,
            start_time=self.start_time,
            duration=self.duration,
            description=self.description,
            status=self.status,
            priority=self.priority,
            metadata=self.metadata,
            tags=combined_tags,
            compliance_level=self.compliance_level
        )
    
    def with_status(self, status: EventStatus) -> 'ImmutableEvent':
        """Pure transformation: update status"""
        return ImmutableEvent(
            uid=self.uid,
            summary=self.summary,
            start_time=self.start_time,
            duration=self.duration,
            description=self.description,
            status=status,
            priority=self.priority,
            metadata=self.metadata,
            tags=self.tags,
            compliance_level=self.compliance_level
        )
    
    def get_metadata_dict(self) -> Dict[str, Any]:
        """Helper: Get metadata as dictionary (read-only view)"""
        return dict(self.metadata)
    
    def calculate_hash(self) -> str:
        """Calculate deterministic hash for deduplication"""
        # Create consistent string representation for hashing
        hash_data = {
            'uid': self.uid,
            'summary': self.summary,
            'start_time': self.start_time.isoformat(),
            'duration': str(self.duration),
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority.value,
            'metadata': dict(self.metadata),
            'tags': sorted(list(self.tags)),
            'compliance_level': self.compliance_level.value
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()

# ===== CALENDAR DATA STRUCTURE =====

@dataclass(frozen=True)
class CalendarData(Transformable):
    """
    Immutable calendar data structure
    
    Represents complete calendar state without mutable operations.
    All calendar modifications return new instances.
    """
    events: Tuple[ImmutableEvent, ...] = field(default_factory=tuple)
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    version: str = "v1"
    created_at: datetime = field(default_factory=datetime.utcnow)
    name: str = ""
    description: str = ""
    compliance_level: ComplianceLevel = ComplianceLevel.BASIC
    
    def __post_init__(self):
        """Validate calendar data constraints"""
        # Ensure all events are immutable
        for i, event in enumerate(self.events):
            if not isinstance(event, ImmutableEvent):
                raise ImmutableValidationError(
                    "CalendarData", 
                    f"events[{i}]", 
                    "non_immutable_event_detected"
                )
        
        # Validate metadata immutability
        validate_immutable_field(dict(self.metadata), "metadata", "CalendarData")
    
    def transform(self, func: Transform['CalendarData', T]) -> T:
        """Apply transformation function to calendar data"""
        return func(self)
    
    def validate_purity(self) -> bool:
        """Validate calendar data maintains immutable constraints"""
        try:
            # Validate all events maintain purity
            for event in self.events:
                if not event.validate_purity():
                    return False
            
            # Validate top-level immutability
            for field_name in self.__dataclass_fields__:
                if field_name == 'events':
                    continue  # Already validated above
                field_value = getattr(self, field_name)
                validate_immutable_field(field_value, field_name, "CalendarData")
            
            return True
        except ImmutableValidationError:
            return False
    
    def add_event(self, event: ImmutableEvent) -> 'CalendarData':
        """Pure transformation: add event"""
        return CalendarData(
            events=self.events + (event,),
            metadata=self.metadata,
            version=self.version,
            created_at=self.created_at,
            name=self.name,
            description=self.description,
            compliance_level=self.compliance_level
        )
    
    def remove_event(self, event_uid: str) -> 'CalendarData':
        """Pure transformation: remove event by UID"""
        filtered_events = tuple(
            event for event in self.events 
            if event.uid != event_uid
        )
        
        return CalendarData(
            events=filtered_events,
            metadata=self.metadata,
            version=self.version,
            created_at=self.created_at,
            name=self.name,
            description=self.description,
            compliance_level=self.compliance_level
        )
    
    def filter_events(self, predicate) -> 'CalendarData':
        """Pure transformation: filter events"""
        filtered_events = tuple(filter(predicate, self.events))
        
        return CalendarData(
            events=filtered_events,
            metadata=self.metadata,
            version=self.version,
            created_at=self.created_at,
            name=self.name,
            description=self.description,
            compliance_level=self.compliance_level
        )
    
    def map_events(self, transform_func) -> 'CalendarData':
        """Pure transformation: map over events"""
        transformed_events = tuple(map(transform_func, self.events))
        
        return CalendarData(
            events=transformed_events,
            metadata=self.metadata,
            version=self.version,
            created_at=self.created_at,
            name=self.name,
            description=self.description,
            compliance_level=self.compliance_level
        )
    
    def with_metadata(self, **new_metadata) -> 'CalendarData':
        """Pure transformation: merge calendar metadata"""
        existing_meta = dict(self.metadata)
        merged_metadata = {**existing_meta, **new_metadata}
        metadata_tuple = tuple(sorted(merged_metadata.items()))
        
        return CalendarData(
            events=self.events,
            metadata=metadata_tuple,
            version=self.version,
            created_at=self.created_at,
            name=self.name,
            description=self.description,
            compliance_level=self.compliance_level
        )
    
    def get_events_by_status(self, status: EventStatus) -> Tuple[ImmutableEvent, ...]:
        """Get events filtered by status (read-only view)"""
        return tuple(event for event in self.events if event.status == status)
    
    def get_events_by_priority(self, priority: PriorityLevel) -> Tuple[ImmutableEvent, ...]:
        """Get events filtered by priority (read-only view)"""
        return tuple(event for event in self.events if event.priority == priority)
    
    def calculate_hash(self) -> str:
        """Calculate deterministic hash for calendar state"""
        hash_data = {
            'events': [event.calculate_hash() for event in self.events],
            'metadata': dict(self.metadata),
            'version': self.version,
            'name': self.name,
            'description': self.description,
            'compliance_level': self.compliance_level.value
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()

# ===== DISTRIBUTION STRUCTURES =====

@dataclass(frozen=True)
class DistributionTarget(Transformable):
    """Immutable distribution target specification"""
    channel: DistributionChannel
    address: str
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    priority: PriorityLevel = PriorityLevel.MEDIUM
    
    def transform(self, func: Transform['DistributionTarget', T]) -> T:
        return func(self)
    
    def validate_purity(self) -> bool:
        try:
            validate_immutable_field(dict(self.metadata), "metadata", "DistributionTarget")
            return True
        except ImmutableValidationError:
            return False

@dataclass(frozen=True)
class DistributionJob(Transformable):
    """Immutable distribution job specification"""
    calendar: CalendarData
    targets: Tuple[DistributionTarget, ...] = field(default_factory=tuple)
    job_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    
    def transform(self, func: Transform['DistributionJob', T]) -> T:
        return func(self)
    
    def validate_purity(self) -> bool:
        try:
            if not self.calendar.validate_purity():
                return False
            for target in self.targets:
                if not target.validate_purity():
                    return False
            validate_immutable_field(dict(self.metadata), "metadata", "DistributionJob")
            return True
        except ImmutableValidationError:
            return False

# ===== AUDIT STRUCTURES =====

@dataclass(frozen=True)
class AuditEvent(Transformable):
    """Immutable audit event for compliance tracking"""
    timestamp: datetime
    event_type: str
    entity_id: str
    entity_type: str
    operation: str
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    compliance_level: ComplianceLevel = ComplianceLevel.BASIC
    
    def transform(self, func: Transform['AuditEvent', T]) -> T:
        return func(self)
    
    def validate_purity(self) -> bool:
        try:
            validate_immutable_field(dict(self.metadata), "metadata", "AuditEvent")
            return True
        except ImmutableValidationError:
            return False

@dataclass(frozen=True)
class ComplianceReport(Transformable):
    """Immutable compliance report structure"""
    report_id: str
    generated_at: datetime
    audit_events: Tuple[AuditEvent, ...] = field(default_factory=tuple)
    compliance_level: ComplianceLevel = ComplianceLevel.BASIC
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    
    def transform(self, func: Transform['ComplianceReport', T]) -> T:
        return func(self)
    
    def validate_purity(self) -> bool:
        try:
            for audit_event in self.audit_events:
                if not audit_event.validate_purity():
                    return False
            validate_immutable_field(dict(self.metadata), "metadata", "ComplianceReport")
            return True
        except ImmutableValidationError:
            return False

# ===== STRUCTURE VALIDATION UTILITIES =====

def validate_structure_hierarchy(structure: Transformable) -> Dict[str, Any]:
    """
    Validate entire structure hierarchy for DOP compliance
    
    Returns comprehensive validation report
    """
    validation_report = {
        "structure_type": type(structure).__name__,
        "is_valid": True,
        "violations": [],
        "warnings": [],
        "nested_validations": []
    }
    
    try:
        # Validate top-level structure
        if not structure.validate_purity():
            validation_report["is_valid"] = False
            validation_report["violations"].append("top_level_purity_violation")
        
        # Recursively validate nested structures
        for field_name, field_value in structure.__dict__.items():
            if isinstance(field_value, Transformable):
                nested_report = validate_structure_hierarchy(field_value)
                validation_report["nested_validations"].append({
                    "field": field_name,
                    "report": nested_report
                })
                
                if not nested_report["is_valid"]:
                    validation_report["is_valid"] = False
                    validation_report["violations"].append(f"nested_violation_in_{field_name}")
            
            elif isinstance(field_value, (tuple, frozenset)) and field_value:
                # Check collections of Transformable objects
                for i, item in enumerate(field_value):
                    if isinstance(item, Transformable):
                        nested_report = validate_structure_hierarchy(item)
                        validation_report["nested_validations"].append({
                            "field": f"{field_name}[{i}]",
                            "report": nested_report
                        })
                        
                        if not nested_report["is_valid"]:
                            validation_report["is_valid"] = False
                            validation_report["violations"].append(
                                f"nested_violation_in_{field_name}[{i}]"
                            )
    
    except Exception as e:
        validation_report["is_valid"] = False
        validation_report["violations"].append(f"validation_exception: {str(e)}")
    
    return validation_report

if __name__ == "__main__":
    # Demonstration of immutable structure usage
    print("=== Pyics Immutable Structure Validation ===")
    
    # Create test event
    test_event = ImmutableEvent(
        uid="test-001",
        summary="Test Event",
        start_time=datetime(2024, 12, 30, 9, 0),
        duration=timedelta(hours=1),
        description="Test event description"
    )
    
    # Test immutability
    modified_event = test_event.with_metadata(location="Conference Room A")
    
    print(f"Original event summary: {test_event.summary}")
    print(f"Modified event metadata: {modified_event.get_metadata_dict()}")
    print(f"Original unchanged: {test_event.get_metadata_dict()}")
    
    # Test calendar creation
    calendar = CalendarData(
        events=(test_event, modified_event),
        name="Test Calendar"
    )
    
    validation_report = validate_structure_hierarchy(calendar)
    print(f"Calendar validation: {validation_report['is_valid']}")
    
    print("=== Structure validation complete ===")

```

## operations.py
```python
#!/usr/bin/env python3
"""
pyics/core/structures/operations.py
Pyics Core Domain Operations: structures

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: structures
Responsibility: Atomic and composed operations on domain data
Compute Weight: Dynamic (varies by operation complexity)

PROBLEM SOLVED: Centralized operation definitions for structures domain
DEPENDENCIES: structures.data_types, structures.relations, typing
THREAD SAFETY: Yes - pure functions with immutable data
DETERMINISTIC: Yes - deterministic operations on immutable data

This module provides atomic and composed operations for the structures
domain, implementing pure functions that transform immutable data structures
following DOP principles.
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator, Any
from functools import reduce, partial
from dataclasses import replace
import logging

# Import domain data types and relations
from .data_types import (
    StructuresEntity,
    StructuresConfig,
    StructuresResult,
    StructuresCollection,
    StructuresIndex,
    StructuresFilter,
    StructuresStatus,
    StructuresPriority
)
from .relations import RelationGraph, Relation, RelationType

logger = logging.getLogger(f"pyics.core.structures.operations")

# Atomic operations (pure functions)
def create_entity(
    entity_id: str,
    name: str,
    status: StructuresStatus = StructuresStatus.INITIALIZED,
    priority: StructuresPriority = StructuresPriority.MEDIUM,
    **metadata
) -> StructuresEntity:
    """
    Create a new structures entity
    
    Pure function for entity creation
    """
    return StructuresEntity(
        id=entity_id,
        name=name,
        status=status,
        priority=priority,
        metadata=metadata
    )

def update_entity_status(
    entity: StructuresEntity,
    new_status: StructuresStatus
) -> StructuresEntity:
    """
    Update entity status (returns new entity)
    
    Pure function for status updates
    """
    return replace(entity, status=new_status)

def update_entity_priority(
    entity: StructuresEntity,
    new_priority: StructuresPriority
) -> StructuresEntity:
    """
    Update entity priority (returns new entity)
    
    Pure function for priority updates
    """
    return replace(entity, priority=new_priority)

def add_entity_metadata(
    entity: StructuresEntity,
    key: str,
    value: Any
) -> StructuresEntity:
    """
    Add metadata to entity (returns new entity)
    
    Pure function for metadata updates
    """
    new_metadata = {**entity.metadata, key: value}
    return replace(entity, metadata=new_metadata)

# Collection operations (pure functions)
def filter_entities_by_status(
    entities: StructuresCollection,
    status: StructuresStatus
) -> StructuresCollection:
    """
    Filter entities by status
    
    Pure filtering function
    """
    return [entity for entity in entities if entity.status == status]

def filter_entities_by_priority(
    entities: StructuresCollection,
    min_priority: StructuresPriority
) -> StructuresCollection:
    """
    Filter entities by minimum priority
    
    Pure filtering function
    """
    return [
        entity for entity in entities 
        if entity.priority.value >= min_priority.value
    ]

def sort_entities_by_priority(
    entities: StructuresCollection,
    descending: bool = True
) -> StructuresCollection:
    """
    Sort entities by priority
    
    Pure sorting function
    """
    return sorted(
        entities,
        key=lambda entity: entity.priority.value,
        reverse=descending
    )

def group_entities_by_status(
    entities: StructuresCollection
) -> Dict[StructuresStatus, StructuresCollection]:
    """
    Group entities by status
    
    Pure grouping function
    """
    groups: Dict[StructuresStatus, StructuresCollection] = {}
    
    for entity in entities:
        if entity.status not in groups:
            groups[entity.status] = []
        groups[entity.status].append(entity)
    
    return groups

# Index operations (pure functions)
def build_entity_index(
    entities: StructuresCollection
) -> StructuresIndex:
    """
    Build index from entity collection
    
    Pure function for index creation
    """
    return {entity.id: entity for entity in entities}

def merge_entity_indices(
    *indices: StructuresIndex
) -> StructuresIndex:
    """
    Merge multiple entity indices
    
    Pure function for index merging
    """
    merged = {}
    for index in indices:
        merged.update(index)
    return merged

def filter_index_by_predicate(
    index: StructuresIndex,
    predicate: Callable[[StructuresEntity], bool]
) -> StructuresIndex:
    """
    Filter index by predicate function
    
    Pure filtering function
    """
    return {
        entity_id: entity 
        for entity_id, entity in index.items() 
        if predicate(entity)
    }

# Composed operations (higher-order functions)
def process_entity_collection(
    entities: StructuresCollection,
    operations: List[Callable[[StructuresEntity], StructuresEntity]]
) -> StructuresCollection:
    """
    Apply a sequence of operations to entity collection
    
    Composed operation using function composition
    """
    def apply_operations(entity: StructuresEntity) -> StructuresEntity:
        return reduce(lambda e, op: op(e), operations, entity)
    
    return [apply_operations(entity) for entity in entities]

def transform_collection_with_config(
    entities: StructuresCollection,
    config: StructuresConfig
) -> StructuresResult:
    """
    Transform collection based on configuration
    
    Composed operation with result wrapping
    """
    try:
        # Apply configuration-based transformations
        filtered_entities = entities[:config.max_items] if config.max_items > 0 else entities
        
        if not config.enabled:
            return StructuresResult(
                success=True,
                data=filtered_entities,
                metadata={"config_enabled": False}
            )
        
        # Process entities based on configuration
        processed_entities = []
        for entity in filtered_entities:
            # Apply configuration-specific processing
            if config.options.get("auto_priority_boost", False):
                entity = update_entity_priority(entity, StructuresPriority.HIGH)
            
            processed_entities.append(entity)
        
        return StructuresResult(
            success=True,
            data=processed_entities,
            metadata={
                "processed_count": len(processed_entities),
                "config_applied": True
            }
        )
    
    except Exception as e:
        logger.error(f"Collection transformation failed: {e}")
        return StructuresResult(
            success=False,
            error_message=str(e),
            metadata={"operation": "transform_collection_with_config"}
        )

def validate_entity_collection(
    entities: StructuresCollection,
    validation_rules: List[Callable[[StructuresEntity], bool]]
) -> StructuresResult:
    """
    Validate entity collection against rules
    
    Composed validation operation
    """
    invalid_entities = []
    
    for entity in entities:
        for rule in validation_rules:
            if not rule(entity):
                invalid_entities.append(entity.id)
                break
    
    success = len(invalid_entities) == 0
    
    return StructuresResult(
        success=success,
        data=entities if success else None,
        error_message=f"Validation failed for entities: {invalid_entities}" if not success else None,
        metadata={
            "validated_count": len(entities),
            "invalid_count": len(invalid_entities),
            "rules_applied": len(validation_rules)
        }
    )

# Relation-based operations
def find_related_entities_by_type(
    graph: RelationGraph,
    entity_id: str,
    relation_type: RelationType
) -> StructuresCollection:
    """
    Find entities related by specific relation type
    
    Pure function combining relations and entities
    """
    relations = [
        rel for rel in graph.relations
        if (rel.source_id == entity_id or rel.target_id == entity_id)
        and rel.relation_type == relation_type
    ]
    
    related_ids = set()
    for rel in relations:
        if rel.source_id == entity_id:
            related_ids.add(rel.target_id)
        else:
            related_ids.add(rel.source_id)
    
    return [
        graph.entity_index[entity_id]
        for entity_id in related_ids
        if entity_id in graph.entity_index
    ]

# Utility functions for operation composition
def compose_operations(*operations: Callable) -> Callable:
    """
    Compose multiple operations into a single function
    
    Functional composition utility
    """
    return reduce(lambda f, g: lambda x: f(g(x)), operations, lambda x: x)

def partial_operation(operation: Callable, **kwargs) -> Callable:
    """
    Create partial operation with fixed parameters
    
    Partial application utility
    """
    return partial(operation, **kwargs)

# Predefined operation sets
STANDARD_ENTITY_OPERATIONS = [
    partial_operation(update_entity_status, new_status=StructuresStatus.PROCESSING),
]

PRIORITY_BOOST_OPERATIONS = [
    partial_operation(update_entity_priority, new_priority=StructuresPriority.HIGH),
    partial_operation(add_entity_metadata, key="priority_boosted", value=True),
]

# Export interface
__all__ = [
    # Atomic operations
    'create_entity',
    'update_entity_status',
    'update_entity_priority',
    'add_entity_metadata',
    
    # Collection operations
    'filter_entities_by_status',
    'filter_entities_by_priority',
    'sort_entities_by_priority',
    'group_entities_by_status',
    
    # Index operations
    'build_entity_index',
    'merge_entity_indices',
    'filter_index_by_predicate',
    
    # Composed operations
    'process_entity_collection',
    'transform_collection_with_config',
    'validate_entity_collection',
    
    # Relation-based operations
    'find_related_entities_by_type',
    
    # Utilities
    'compose_operations',
    'partial_operation',
    
    # Predefined operations
    'STANDARD_ENTITY_OPERATIONS',
    'PRIORITY_BOOST_OPERATIONS',
]

# [EOF] - End of structures operations.py module

```

## relations.py
```python
#!/usr/bin/env python3
"""
pyics/core/structures/relations.py
Pyics Core Domain Relations: structures

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: structures
Responsibility: Structural mappings and relationship definitions
Compute Weight: Static to Computed (depending on relation complexity)

PROBLEM SOLVED: Centralized relationship mapping for structures domain
DEPENDENCIES: structures.data_types, typing, dataclasses
THREAD SAFETY: Yes - immutable relation structures
DETERMINISTIC: Yes - static relationship definitions

This module defines structural relationships and mappings between entities
in the structures domain, following DOP principles with immutable
relation containers and pure transformation functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator
from enum import Enum, auto

# Import domain data types
from .data_types import (
    StructuresEntity,
    StructuresCollection,
    StructuresIndex,
    StructuresFilter
)

# Relationship types
class RelationType(Enum):
    """Types of relationships in structures domain"""
    ONE_TO_ONE = auto()
    ONE_TO_MANY = auto()
    MANY_TO_MANY = auto()
    HIERARCHICAL = auto()
    DEPENDENCY = auto()

class RelationStrength(Enum):
    """Strength of relationships"""
    WEAK = auto()
    STRONG = auto()
    CRITICAL = auto()

# Relation containers
@dataclass(frozen=True)
class Relation:
    """
    Immutable relation between structures entities
    
    Defines structural relationship with metadata
    """
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: RelationStrength = RelationStrength.WEAK
    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass(frozen=True)
class RelationGraph:
    """
    Immutable graph of relations in structures domain
    
    Container for complete relationship structure
    """
    relations: Tuple[Relation, ...] = field(default_factory=tuple)
    entity_index: Dict[str, StructuresEntity] = field(default_factory=dict)
    
    def get_relations_for_entity(self, entity_id: str) -> List[Relation]:
        """Get all relations involving an entity"""
        return [
            rel for rel in self.relations 
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
    
    def get_related_entities(self, entity_id: str) -> List[StructuresEntity]:
        """Get all entities related to a given entity"""
        relations = self.get_relations_for_entity(entity_id)
        related_ids = set()
        
        for rel in relations:
            if rel.source_id == entity_id:
                related_ids.add(rel.target_id)
            else:
                related_ids.add(rel.source_id)
        
        return [
            self.entity_index[entity_id] 
            for entity_id in related_ids 
            if entity_id in self.entity_index
        ]

@dataclass(frozen=True)
class RelationMapping:
    """
    Static mapping configuration for structures relations
    
    Defines how entities should be related
    """
    mapping_name: str
    source_type: str
    target_type: str
    relation_type: RelationType
    mapping_function: Optional[str] = None  # Function name for dynamic mapping
    validation_rules: Tuple[str, ...] = field(default_factory=tuple)

# Relation building functions (pure functions)
def create_relation(
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    strength: RelationStrength = RelationStrength.WEAK,
    **metadata
) -> Relation:
    """
    Create a new relation between entities
    
    Pure function for relation creation
    """
    return Relation(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        strength=strength,
        metadata=metadata
    )

def build_relation_graph(
    entities: StructuresCollection,
    relations: List[Relation]
) -> RelationGraph:
    """
    Build relation graph from entities and relations
    
    Pure function for graph construction
    """
    entity_index = {entity.id: entity for entity in entities}
    
    return RelationGraph(
        relations=tuple(relations),
        entity_index=entity_index
    )

def filter_relations_by_type(
    graph: RelationGraph,
    relation_type: RelationType
) -> List[Relation]:
    """
    Filter relations by type
    
    Pure filtering function
    """
    return [
        rel for rel in graph.relations 
        if rel.relation_type == relation_type
    ]

def find_relation_path(
    graph: RelationGraph,
    source_id: str,
    target_id: str,
    max_depth: int = 5
) -> Optional[List[str]]:
    """
    Find path between entities through relations
    
    Pure pathfinding function with depth limit
    """
    if source_id == target_id:
        return [source_id]
    
    visited = set()
    queue = [(source_id, [source_id])]
    
    while queue:
        current_id, path = queue.pop(0)
        
        if len(path) > max_depth:
            continue
            
        if current_id in visited:
            continue
            
        visited.add(current_id)
        
        for rel in graph.get_relations_for_entity(current_id):
            next_id = rel.target_id if rel.source_id == current_id else rel.source_id
            
            if next_id == target_id:
                return path + [next_id]
            
            if next_id not in visited:
                queue.append((next_id, path + [next_id]))
    
    return None

# Predefined relation mappings for structures domain
DEFAULT_RELATION_MAPPINGS: List[RelationMapping] = [
    RelationMapping(
        mapping_name="hierarchical_parent_child",
        source_type="StructuresEntity",
        target_type="StructuresEntity",
        relation_type=RelationType.HIERARCHICAL,
        validation_rules=("source_id != target_id", "no_circular_dependencies")
    ),
    RelationMapping(
        mapping_name="dependency_chain",
        source_type="StructuresEntity",
        target_type="StructuresEntity",
        relation_type=RelationType.DEPENDENCY,
        validation_rules=("source_id != target_id", "acyclic_dependencies")
    ),
]

# Export interface
__all__ = [
    'RelationType',
    'RelationStrength',
    'Relation',
    'RelationGraph',
    'RelationMapping',
    'create_relation',
    'build_relation_graph',
    'filter_relations_by_type',
    'find_relation_path',
    'DEFAULT_RELATION_MAPPINGS',
]

# [EOF] - End of structures relations.py module

```

## __init__.py
```python
#!/usr/bin/env python3
"""
pyics/core/structures/implementations/__init__.py
Domain Implementation Module

Contains concrete implementations for structures domain.
All implementations must follow linear architecture principles.
"""

# Domain implementation exports
__all__ = []

# Implementation registry function
def get_domain_exports():
    """Export all domain implementations for registration"""
    return {}

```

