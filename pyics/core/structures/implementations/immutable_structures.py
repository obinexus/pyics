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
