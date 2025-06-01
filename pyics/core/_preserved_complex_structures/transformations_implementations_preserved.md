# Preserved Complex Structure: transformations/implementations

Preservation timestamp: 2025-06-01T23:50:03.918614
Systematic cleanup phase: structure_flattening

## data_types.py
```python
#!/usr/bin/env python3
"""
pyics/core/transformations/data_types.py
Pyics Core Domain Data Types: transformations

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: transformations
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for transformations domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the transformations
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class TransformationsStatus(Enum):
    """Status enumeration for transformations domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class TransformationsPriority(Enum):
    """Priority levels for transformations domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class TransformationsEntity:
    """
    Base entity for transformations domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: TransformationsStatus = TransformationsStatus.INITIALIZED
    priority: TransformationsPriority = TransformationsPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TransformationsConfig:
    """
    Configuration data structure for transformations domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TransformationsResult:
    """
    Result container for transformations domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class TransformationsProcessor(Protocol):
    """Protocol for transformations domain processors"""
    
    def process(self, entity: TransformationsEntity) -> TransformationsResult:
        """Process a transformations entity"""
        ...
    
    def validate(self, entity: TransformationsEntity) -> bool:
        """Validate a transformations entity"""
        ...

class TransformationsRepository(Protocol):
    """Protocol for transformations domain data repositories"""
    
    def store(self, entity: TransformationsEntity) -> bool:
        """Store a transformations entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[TransformationsEntity]:
        """Retrieve a transformations entity by ID"""
        ...
    
    def list_all(self) -> List[TransformationsEntity]:
        """List all transformations entities"""
        ...

# Type aliases for complex structures
TransformationsCollection = List[TransformationsEntity]
TransformationsIndex = Dict[str, TransformationsEntity]
TransformationsFilter = Dict[str, Any]

# Export interface
__all__ = [
    'TransformationsStatus',
    'TransformationsPriority',
    'TransformationsEntity',
    'TransformationsConfig',
    'TransformationsResult',
    'TransformationsProcessor',
    'TransformationsRepository',
    'TransformationsCollection',
    'TransformationsIndex',
    'TransformationsFilter',
]

# [EOF] - End of transformations data_types.py module

```

## operations.py
```python
#!/usr/bin/env python3
"""
pyics/core/transformations/operations.py
Pyics Core Domain Operations: transformations

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: transformations
Responsibility: Atomic and composed operations on domain data
Compute Weight: Dynamic (varies by operation complexity)

PROBLEM SOLVED: Centralized operation definitions for transformations domain
DEPENDENCIES: transformations.data_types, transformations.relations, typing
THREAD SAFETY: Yes - pure functions with immutable data
DETERMINISTIC: Yes - deterministic operations on immutable data

This module provides atomic and composed operations for the transformations
domain, implementing pure functions that transform immutable data structures
following DOP principles.
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator, Any
from functools import reduce, partial
from dataclasses import replace
import logging

# Import domain data types and relations
from .data_types import (
    TransformationsEntity,
    TransformationsConfig,
    TransformationsResult,
    TransformationsCollection,
    TransformationsIndex,
    TransformationsFilter,
    TransformationsStatus,
    TransformationsPriority
)
from .relations import RelationGraph, Relation, RelationType

logger = logging.getLogger(f"pyics.core.transformations.operations")

# Atomic operations (pure functions)
def create_entity(
    entity_id: str,
    name: str,
    status: TransformationsStatus = TransformationsStatus.INITIALIZED,
    priority: TransformationsPriority = TransformationsPriority.MEDIUM,
    **metadata
) -> TransformationsEntity:
    """
    Create a new transformations entity
    
    Pure function for entity creation
    """
    return TransformationsEntity(
        id=entity_id,
        name=name,
        status=status,
        priority=priority,
        metadata=metadata
    )

def update_entity_status(
    entity: TransformationsEntity,
    new_status: TransformationsStatus
) -> TransformationsEntity:
    """
    Update entity status (returns new entity)
    
    Pure function for status updates
    """
    return replace(entity, status=new_status)

def update_entity_priority(
    entity: TransformationsEntity,
    new_priority: TransformationsPriority
) -> TransformationsEntity:
    """
    Update entity priority (returns new entity)
    
    Pure function for priority updates
    """
    return replace(entity, priority=new_priority)

def add_entity_metadata(
    entity: TransformationsEntity,
    key: str,
    value: Any
) -> TransformationsEntity:
    """
    Add metadata to entity (returns new entity)
    
    Pure function for metadata updates
    """
    new_metadata = {**entity.metadata, key: value}
    return replace(entity, metadata=new_metadata)

# Collection operations (pure functions)
def filter_entities_by_status(
    entities: TransformationsCollection,
    status: TransformationsStatus
) -> TransformationsCollection:
    """
    Filter entities by status
    
    Pure filtering function
    """
    return [entity for entity in entities if entity.status == status]

def filter_entities_by_priority(
    entities: TransformationsCollection,
    min_priority: TransformationsPriority
) -> TransformationsCollection:
    """
    Filter entities by minimum priority
    
    Pure filtering function
    """
    return [
        entity for entity in entities 
        if entity.priority.value >= min_priority.value
    ]

def sort_entities_by_priority(
    entities: TransformationsCollection,
    descending: bool = True
) -> TransformationsCollection:
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
    entities: TransformationsCollection
) -> Dict[TransformationsStatus, TransformationsCollection]:
    """
    Group entities by status
    
    Pure grouping function
    """
    groups: Dict[TransformationsStatus, TransformationsCollection] = {}
    
    for entity in entities:
        if entity.status not in groups:
            groups[entity.status] = []
        groups[entity.status].append(entity)
    
    return groups

# Index operations (pure functions)
def build_entity_index(
    entities: TransformationsCollection
) -> TransformationsIndex:
    """
    Build index from entity collection
    
    Pure function for index creation
    """
    return {entity.id: entity for entity in entities}

def merge_entity_indices(
    *indices: TransformationsIndex
) -> TransformationsIndex:
    """
    Merge multiple entity indices
    
    Pure function for index merging
    """
    merged = {}
    for index in indices:
        merged.update(index)
    return merged

def filter_index_by_predicate(
    index: TransformationsIndex,
    predicate: Callable[[TransformationsEntity], bool]
) -> TransformationsIndex:
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
    entities: TransformationsCollection,
    operations: List[Callable[[TransformationsEntity], TransformationsEntity]]
) -> TransformationsCollection:
    """
    Apply a sequence of operations to entity collection
    
    Composed operation using function composition
    """
    def apply_operations(entity: TransformationsEntity) -> TransformationsEntity:
        return reduce(lambda e, op: op(e), operations, entity)
    
    return [apply_operations(entity) for entity in entities]

def transform_collection_with_config(
    entities: TransformationsCollection,
    config: TransformationsConfig
) -> TransformationsResult:
    """
    Transform collection based on configuration
    
    Composed operation with result wrapping
    """
    try:
        # Apply configuration-based transformations
        filtered_entities = entities[:config.max_items] if config.max_items > 0 else entities
        
        if not config.enabled:
            return TransformationsResult(
                success=True,
                data=filtered_entities,
                metadata={"config_enabled": False}
            )
        
        # Process entities based on configuration
        processed_entities = []
        for entity in filtered_entities:
            # Apply configuration-specific processing
            if config.options.get("auto_priority_boost", False):
                entity = update_entity_priority(entity, TransformationsPriority.HIGH)
            
            processed_entities.append(entity)
        
        return TransformationsResult(
            success=True,
            data=processed_entities,
            metadata={
                "processed_count": len(processed_entities),
                "config_applied": True
            }
        )
    
    except Exception as e:
        logger.error(f"Collection transformation failed: {e}")
        return TransformationsResult(
            success=False,
            error_message=str(e),
            metadata={"operation": "transform_collection_with_config"}
        )

def validate_entity_collection(
    entities: TransformationsCollection,
    validation_rules: List[Callable[[TransformationsEntity], bool]]
) -> TransformationsResult:
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
    
    return TransformationsResult(
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
) -> TransformationsCollection:
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
    partial_operation(update_entity_status, new_status=TransformationsStatus.PROCESSING),
]

PRIORITY_BOOST_OPERATIONS = [
    partial_operation(update_entity_priority, new_priority=TransformationsPriority.HIGH),
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

# [EOF] - End of transformations operations.py module

```

## pure_transforms.py
```python
#!/usr/bin/env python3
"""
pyics/core/transforms.py
Pure Transformation Library for Data-Oriented Programming

Stateless transformation utilities that operate on immutable structures.
All business logic MUST route through these registered transformations.

Zero Trust Principle: No state mutation allowed - only pure transformations.

Author: OBINexus Engineering Team / Nnamdi Okpala
License: MIT
Compliance: DOP Canon Phase 3.1
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import re
from functools import partial

from .lambda import (
    Lambda, Transform, Predicate, Aggregator, 
    pure_function, register_transform, GLOBAL_TRANSFORM_REGISTRY
)
from .structures import (
    ImmutableEvent, CalendarData, EventStatus, PriorityLevel, 
    ComplianceLevel, DistributionTarget, DistributionChannel,
    AuditEvent, ComplianceReport
)

# ===== EVENT TRANSFORMATIONS =====

@register_transform("add_timezone_metadata", version="core")
@pure_function
def add_timezone_metadata(timezone: str) -> Transform[ImmutableEvent, ImmutableEvent]:
    """Create transformation to add timezone metadata to events"""
    def transform_event(event: ImmutableEvent) -> ImmutableEvent:
        return event.with_metadata(timezone=timezone)
    return transform_event

@register_transform("shift_event_time", version="core")
@pure_function
def shift_event_time(delta: timedelta) -> Transform[ImmutableEvent, ImmutableEvent]:
    """Create transformation to shift event start time"""
    def transform_event(event: ImmutableEvent) -> ImmutableEvent:
        new_start = event.start_time + delta
        return ImmutableEvent(
            uid=event.uid,
            summary=event.summary,
            start_time=new_start,
            duration=event.duration,
            description=event.description,
            status=event.status,
            priority=event.priority,
            metadata=event.metadata,
            tags=event.tags,
            compliance_level=event.compliance_level
        )
    return transform_event

@register_transform("set_event_priority", version="core")
@pure_function
def set_event_priority(priority: PriorityLevel) -> Transform[ImmutableEvent, ImmutableEvent]:
    """Create transformation to set event priority"""
    def transform_event(event: ImmutableEvent) -> ImmutableEvent:
        return ImmutableEvent(
            uid=event.uid,
            summary=event.summary,
            start_time=event.start_time,
            duration=event.duration,
            description=event.description,
            status=event.status,
            priority=priority,
            metadata=event.metadata,
            tags=event.tags,
            compliance_level=event.compliance_level
        )
    return transform_event

@register_transform("update_event_status", version="core")
@pure_function
def update_event_status(status: EventStatus) -> Transform[ImmutableEvent, ImmutableEvent]:
    """Create transformation to update event status"""
    def transform_event(event: ImmutableEvent) -> ImmutableEvent:
        return event.with_status(status)
    return transform_event

@register_transform("add_event_tags", version="core")
@pure_function
def add_event_tags(*tags: str) -> Transform[ImmutableEvent, ImmutableEvent]:
    """Create transformation to add tags to events"""
    def transform_event(event: ImmutableEvent) -> ImmutableEvent:
        return event.with_tags(*tags)
    return transform_event

@register_transform("sanitize_event_description", version="core")
@pure_function
def sanitize_event_description(event: ImmutableEvent) -> ImmutableEvent:
    """Sanitize event description for security compliance"""
    # Remove potentially dangerous content
    sanitized_description = re.sub(r'<[^>]+>', '', event.description)  # Remove HTML tags
    sanitized_description = re.sub(r'[^\w\s\-.,!?]', '', sanitized_description)  # Remove special chars
    
    return ImmutableEvent(
        uid=event.uid,
        summary=event.summary,
        start_time=event.start_time,
        duration=event.duration,
        description=sanitized_description.strip(),
        status=event.status,
        priority=event.priority,
        metadata=event.metadata,
        tags=event.tags,
        compliance_level=event.compliance_level
    )

# ===== VALIDATION PREDICATES =====

@register_transform("validate_event_duration", version="core")
@pure_function
def validate_event_duration(min_duration: timedelta) -> Predicate[ImmutableEvent]:
    """Create validation predicate for minimum event duration"""
    def validate(event: ImmutableEvent) -> bool:
        return event.duration >= min_duration
    return validate

@register_transform("validate_event_in_future", version="core")
@pure_function
def validate_event_in_future(reference_time: Optional[datetime] = None) -> Predicate[ImmutableEvent]:
    """Create validation predicate for events scheduled in the future"""
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    def validate(event: ImmutableEvent) -> bool:
        return event.start_time > reference_time
    return validate

@register_transform("validate_event_priority", version="core")
@pure_function
def validate_event_priority(min_priority: PriorityLevel) -> Predicate[ImmutableEvent]:
    """Create validation predicate for minimum event priority"""
    def validate(event: ImmutableEvent) -> bool:
        return event.priority.value >= min_priority.value
    return validate

@register_transform("validate_event_compliance", version="core")
@pure_function
def validate_event_compliance(required_level: ComplianceLevel) -> Predicate[ImmutableEvent]:
    """Create validation predicate for compliance level requirements"""
    def validate(event: ImmutableEvent) -> bool:
        compliance_hierarchy = {
            ComplianceLevel.BASIC: 1,
            ComplianceLevel.STANDARD: 2,
            ComplianceLevel.ENHANCED: 3,
            ComplianceLevel.ENTERPRISE: 4,
            ComplianceLevel.REGULATORY: 5
        }
        return compliance_hierarchy[event.compliance_level] >= compliance_hierarchy[required_level]
    return validate

# ===== CALENDAR TRANSFORMATIONS =====

@register_transform("filter_calendar_by_status", version="core")
@pure_function
def filter_calendar_by_status(status: EventStatus) -> Transform[CalendarData, CalendarData]:
    """Create transformation to filter calendar events by status"""
    def transform_calendar(calendar: CalendarData) -> CalendarData:
        return calendar.filter_events(lambda event: event.status == status)
    return transform_calendar

@register_transform("filter_calendar_by_priority", version="core")
@pure_function
def filter_calendar_by_priority(min_priority: PriorityLevel) -> Transform[CalendarData, CalendarData]:
    """Create transformation to filter calendar events by minimum priority"""
    def transform_calendar(calendar: CalendarData) -> CalendarData:
        predicate = validate_event_priority(min_priority)
        return calendar.filter_events(predicate)
    return transform_calendar

@register_transform("sort_calendar_by_time", version="core")
@pure_function
def sort_calendar_by_time(calendar: CalendarData) -> CalendarData:
    """Sort calendar events by start time"""
    sorted_events = tuple(sorted(calendar.events, key=lambda e: e.start_time))
    return CalendarData(
        events=sorted_events,
        metadata=calendar.metadata,
        version=calendar.version,
        created_at=calendar.created_at,
        name=calendar.name,
        description=calendar.description,
        compliance_level=calendar.compliance_level
    )

@register_transform("deduplicate_calendar_events", version="core")
@pure_function
def deduplicate_calendar_events(calendar: CalendarData) -> CalendarData:
    """Remove duplicate events from calendar based on content hash"""
    seen_hashes = set()
    unique_events = []
    
    for event in calendar.events:
        event_hash = event.calculate_hash()
        if event_hash not in seen_hashes:
            seen_hashes.add(event_hash)
            unique_events.append(event)
    
    return CalendarData(
        events=tuple(unique_events),
        metadata=calendar.metadata,
        version=calendar.version,
        created_at=calendar.created_at,
        name=calendar.name,
        description=calendar.description,
        compliance_level=calendar.compliance_level
    )

@register_transform("apply_calendar_compliance", version="core")
@pure_function
def apply_calendar_compliance(compliance_level: ComplianceLevel) -> Transform[CalendarData, CalendarData]:
    """Apply compliance level to calendar and all events"""
    def transform_calendar(calendar: CalendarData) -> CalendarData:
        # Update compliance level for all events
        compliance_events = []
        for event in calendar.events:
            updated_event = ImmutableEvent(
                uid=event.uid,
                summary=event.summary,
                start_time=event.start_time,
                duration=event.duration,
                description=event.description,
                status=event.status,
                priority=event.priority,
                metadata=event.metadata,
                tags=event.tags,
                compliance_level=compliance_level
            )
            compliance_events.append(updated_event)
        
        return CalendarData(
            events=tuple(compliance_events),
            metadata=calendar.metadata,
            version=calendar.version,
            created_at=calendar.created_at,
            name=calendar.name,
            description=calendar.description,
            compliance_level=compliance_level
        )
    return transform_calendar

# ===== AGGREGATION TRANSFORMATIONS =====

@register_transform("aggregate_events_by_date", version="core")
@pure_function
def aggregate_events_by_date() -> Aggregator[ImmutableEvent, Dict[str, List[ImmutableEvent]]]:
    """Aggregate events by date"""
    def aggregate(events: List[ImmutableEvent]) -> Dict[str, List[ImmutableEvent]]:
        result = {}
        for event in events:
            date_key = event.start_time.date().isoformat()
            if date_key not in result:
                result[date_key] = []
            result[date_key].append(event)
        return result
    return aggregate

@register_transform("aggregate_events_by_priority", version="core")
@pure_function
def aggregate_events_by_priority() -> Aggregator[ImmutableEvent, Dict[PriorityLevel, List[ImmutableEvent]]]:
    """Aggregate events by priority level"""
    def aggregate(events: List[ImmutableEvent]) -> Dict[PriorityLevel, List[ImmutableEvent]]:
        result = {}
        for event in events:
            if event.priority not in result:
                result[event.priority] = []
            result[event.priority].append(event)
        return result
    return aggregate

@register_transform("aggregate_events_by_status", version="core")
@pure_function
def aggregate_events_by_status() -> Aggregator[ImmutableEvent, Dict[EventStatus, List[ImmutableEvent]]]:
    """Aggregate events by status"""
    def aggregate(events: List[ImmutableEvent]) -> Dict[EventStatus, List[ImmutableEvent]]:
        result = {}
        for event in events:
            if event.status not in result:
                result[event.status] = []
            result[event.status].append(event)
        return result
    return aggregate

# ===== FORMAT TRANSFORMATIONS =====

@register_transform("format_event_as_ics", version="core")
@pure_function
def format_event_as_ics(event: ImmutableEvent) -> str:
    """Transform event to ICS format"""
    # Basic ICS event formatting
    ics_lines = [
        "BEGIN:VEVENT",
        f"UID:{event.uid}",
        f"SUMMARY:{event.summary}",
        f"DTSTART:{event.start_time.strftime('%Y%m%dT%H%M%SZ')}",
        f"DURATION:PT{int(event.duration.total_seconds())}S",
        f"DESCRIPTION:{event.description}",
        f"STATUS:{event.status.value.upper()}",
        f"PRIORITY:{event.priority.value}"
    ]
    
    # Add metadata as custom properties
    for key, value in event.get_metadata_dict().items():
        ics_lines.append(f"X-PYICS-{key.upper()}:{value}")
    
    # Add tags
    if event.tags:
        ics_lines.append(f"CATEGORIES:{','.join(sorted(event.tags))}")
    
    ics_lines.append("END:VEVENT")
    return "\n".join(ics_lines)

@register_transform("format_calendar_as_ics", version="core")
@pure_function
def format_calendar_as_ics(calendar: CalendarData) -> str:
    """Transform calendar to complete ICS format"""
    ics_lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        f"PRODID:-//Pyics {calendar.version}//EN"
    ]
    
    # Add calendar metadata
    if calendar.name:
        ics_lines.append(f"X-WR-CALNAME:{calendar.name}")
    
    if calendar.description:
        ics_lines.append(f"X-WR-CALDESC:{calendar.description}")
    
    # Add compliance level
    ics_lines.append(f"X-PYICS-COMPLIANCE:{calendar.compliance_level.value}")
    
    # Add all events
    for event in calendar.events:
        event_ics = format_event_as_ics(event)
        ics_lines.append(event_ics)
    
    ics_lines.append("END:VCALENDAR")
    return "\n".join(ics_lines)

# ===== MILESTONE TRACKING TRANSFORMATIONS =====

@register_transform("create_milestone_series", version="core")
@pure_function
def create_milestone_series(
    start_date: datetime,
    milestones: List[str],
    interval_days: int = 14,
    priority: PriorityLevel = PriorityLevel.HIGH,
    compliance_level: ComplianceLevel = ComplianceLevel.ENHANCED
) -> List[ImmutableEvent]:
    """Create series of milestone events"""
    events = []
    for i, milestone in enumerate(milestones):
        event_time = start_date + timedelta(days=i * interval_days)
        event = ImmutableEvent(
            uid=f"milestone-{i+1:03d}",
            summary=milestone,
            start_time=event_time,
            duration=timedelta(hours=1),
            description=f"Milestone tracking: {milestone}",
            status=EventStatus.SCHEDULED,
            priority=priority,
            compliance_level=compliance_level,
            metadata=(
                ("milestone_index", i),
                ("milestone_series", True),
                ("interval_days", interval_days),
            )
        )
        events.append(event)
    
    return events

@register_transform("create_penalty_escalation_series", version="core")
@pure_function
def create_penalty_escalation_series(
    start_date: datetime,
    penalties: List[Tuple[str, str]],  # (description, amount)
    interval_days: int = 14,
    compliance_level: ComplianceLevel = ComplianceLevel.REGULATORY
) -> List[ImmutableEvent]:
    """Create escalating penalty event series"""
    events = []
    for i, (penalty_desc, penalty_amount) in enumerate(penalties):
        event_time = start_date + timedelta(days=i * interval_days)
        priority = PriorityLevel.CRITICAL if i >= 3 else PriorityLevel.HIGH
        
        event = ImmutableEvent(
            uid=f"penalty-{i+1:03d}",
            summary=f"Civil Collapse Penalty: {penalty_desc}",
            start_time=event_time,
            duration=timedelta(hours=1),
            description=f"Penalty escalation: {penalty_desc} - {penalty_amount}",
            status=EventStatus.SCHEDULED,
            priority=priority,
            compliance_level=compliance_level,
            metadata=(
                ("penalty_amount", penalty_amount),
                ("escalation_level", i + 1),
                ("penalty_series", True),
                ("interval_days", interval_days),
            ),
            tags=frozenset(["penalty", "escalation", "compliance", "legal"])
        )
        events.append(event)
    
    return events

# ===== AUDIT TRANSFORMATIONS =====

@register_transform("create_audit_event", version="core")
@pure_function
def create_audit_event(
    operation: str,
    entity_id: str,
    entity_type: str,
    compliance_level: ComplianceLevel = ComplianceLevel.STANDARD,
    **metadata
) -> AuditEvent:
    """Create audit event for compliance tracking"""
    return AuditEvent(
        timestamp=datetime.utcnow(),
        event_type="system_operation",
        entity_id=entity_id,
        entity_type=entity_type,
        operation=operation,
        metadata=tuple(sorted(metadata.items())),
        compliance_level=compliance_level
    )

@register_transform("aggregate_audit_events", version="core")
@pure_function
def aggregate_audit_events(
    audit_events: List[AuditEvent],
    report_id: str,
    compliance_level: ComplianceLevel = ComplianceLevel.STANDARD
) -> ComplianceReport:
    """Aggregate audit events into compliance report"""
    return ComplianceReport(
        report_id=report_id,
        generated_at=datetime.utcnow(),
        audit_events=tuple(audit_events),
        compliance_level=compliance_level,
        metadata=(
            ("event_count", len(audit_events)),
            ("generation_source", "pyics_core_transforms"),
        )
    )

# ===== COMPOSITION UTILITIES =====

def create_event_processing_pipeline(
    *transform_names: str,
    version: str = "core"
) -> Callable[[ImmutableEvent], ImmutableEvent]:
    """Create event processing pipeline from registered transforms"""
    return GLOBAL_TRANSFORM_REGISTRY.create_pipeline(*transform_names, version=version)

def create_calendar_processing_pipeline(
    *transform_names: str,
    version: str = "core"
) -> Callable[[CalendarData], CalendarData]:
    """Create calendar processing pipeline from registered transforms"""
    return GLOBAL_TRANSFORM_REGISTRY.create_pipeline(*transform_names, version=version)

# ===== VALIDATION PIPELINE =====

@register_transform("validate_complete_event", version="core")
@pure_function
def validate_complete_event(event: ImmutableEvent) -> bool:
    """Complete event validation pipeline"""
    validations = [
        validate_event_duration(timedelta(minutes=1)),
        validate_event_in_future(),
        lambda e: bool(e.uid.strip()),
        lambda e: bool(e.summary.strip()),
        lambda e: e.validate_purity()
    ]
    
    return all(validation(event) for validation in validations)

@register_transform("validate_complete_calendar", version="core")
@pure_function
def validate_complete_calendar(calendar: CalendarData) -> bool:
    """Complete calendar validation pipeline"""
    # Validate calendar structure
    if not calendar.validate_purity():
        return False
    
    # Validate all events
    for event in calendar.events:
        if not validate_complete_event(event):
            return False
    
    # Check for duplicate UIDs
    uids = [event.uid for event in calendar.events]
    if len(uids) != len(set(uids)):
        return False
    
    return True

if __name__ == "__main__":
    # Demonstration of transformation pipeline usage
    print("=== Pyics Core Transformations Demo ===")
    
    # Create test event
    test_event = ImmutableEvent(
        uid="demo-001",
        summary="Demo Event",
        start_time=datetime(2024, 12, 30, 9, 0),
        duration=timedelta(hours=1),
        description="Demonstration event"
    )
    
    # Apply transformations
    timezone_transform = add_timezone_metadata("UTC")
    priority_transform = set_event_priority(PriorityLevel.HIGH)
    
    # Compose transformations
    pipeline = Lambda.compose(priority_transform, timezone_transform)
    transformed_event = pipeline(test_event)
    
    print(f"Original priority: {test_event.priority}")
    print(f"Transformed priority: {transformed_event.priority}")
    print(f"Added metadata: {transformed_event.get_metadata_dict()}")
    
    # Test calendar transformation
    calendar = CalendarData(events=(transformed_event,), name="Demo Calendar")
    
    # Format as ICS
    ics_content = format_calendar_as_ics(calendar)
    print(f"ICS format length: {len(ics_content)} characters")
    
    # Validate transformations
    print(f"Event validation: {validate_complete_event(transformed_event)}")
    print(f"Calendar validation: {validate_complete_calendar(calendar)}")
    
    print("=== Transformation demo complete ===")

```

## relations.py
```python
#!/usr/bin/env python3
"""
pyics/core/transformations/relations.py
Pyics Core Domain Relations: transformations

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: transformations
Responsibility: Structural mappings and relationship definitions
Compute Weight: Static to Computed (depending on relation complexity)

PROBLEM SOLVED: Centralized relationship mapping for transformations domain
DEPENDENCIES: transformations.data_types, typing, dataclasses
THREAD SAFETY: Yes - immutable relation structures
DETERMINISTIC: Yes - static relationship definitions

This module defines structural relationships and mappings between entities
in the transformations domain, following DOP principles with immutable
relation containers and pure transformation functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator
from enum import Enum, auto

# Import domain data types
from .data_types import (
    TransformationsEntity,
    TransformationsCollection,
    TransformationsIndex,
    TransformationsFilter
)

# Relationship types
class RelationType(Enum):
    """Types of relationships in transformations domain"""
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
    Immutable relation between transformations entities
    
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
    Immutable graph of relations in transformations domain
    
    Container for complete relationship structure
    """
    relations: Tuple[Relation, ...] = field(default_factory=tuple)
    entity_index: Dict[str, TransformationsEntity] = field(default_factory=dict)
    
    def get_relations_for_entity(self, entity_id: str) -> List[Relation]:
        """Get all relations involving an entity"""
        return [
            rel for rel in self.relations 
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
    
    def get_related_entities(self, entity_id: str) -> List[TransformationsEntity]:
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
    Static mapping configuration for transformations relations
    
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
    entities: TransformationsCollection,
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

# Predefined relation mappings for transformations domain
DEFAULT_RELATION_MAPPINGS: List[RelationMapping] = [
    RelationMapping(
        mapping_name="hierarchical_parent_child",
        source_type="TransformationsEntity",
        target_type="TransformationsEntity",
        relation_type=RelationType.HIERARCHICAL,
        validation_rules=("source_id != target_id", "no_circular_dependencies")
    ),
    RelationMapping(
        mapping_name="dependency_chain",
        source_type="TransformationsEntity",
        target_type="TransformationsEntity",
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

# [EOF] - End of transformations relations.py module

```

## __init__.py
```python

```

