#!/usr/bin/env python3
"""
refactored/core/structures/__init__.py
RIFT Structures Domain - Immutable Data Containers (Priority Index: 2)

PROBLEM SOLVED: Immutable data container definitions ensuring zero-mutation state 
management across calendar operations
SEPARATION RATIONALE: Data structure definitions require isolation from transformation 
logic to maintain immutability guarantees and enable independent validation.

COST FUNCTION:
- priority_index: 2 (high priority - data foundation)
- compute_time_weight: 0.2 (dataclass instantiation and validation overhead)
- exposure_type: version_required (core data structures used by all versions)
- merge_potential: PRESERVE (data integrity requires dedicated domain)

Author: OBINexus Engineering Team / Nnamdi Okpala
Architecture: Single-Pass RIFT System with Cost-Aware Loading
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, FrozenSet, Callable
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json

# Import protocol dependencies (dependency level 1)
from ..protocols import RIFTTransformable, RIFTImmutable, RIFTValidatable, RIFTEventProtocol, RIFTCalendarProtocol

T = Any

# Domain metadata for cost-aware loading
__domain_metadata__ = {
    "name": "structures",
    "priority_index": 2,
    "compute_time_weight": 0.2,
    "exposure_type": "version_required",
    "dependency_level": 2,
    "thread_safe": True,
    "load_order": 3
}

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

# ===== VALIDATION INFRASTRUCTURE =====

class ImmutableValidationError(Exception):
    """Raised when immutable data structure validation fails"""
    def __init__(self, structure_type: str, field: str, violation: str):
        self.structure_type = structure_type
        self.field = field
        self.violation = violation
        super().__init__(f"Validation failed in {structure_type}.{field}: {violation}")

def validate_immutable_field(value: Any, field_name: str, structure_type: str) -> None:
    """
    Validate field constraints for immutable structures
    
    Ensures no mutable containers can break immutability guarantees.
    
    Args:
        value: Field value to validate
        field_name: Name of field being validated
        structure_type: Type of structure containing field
        
    Raises:
        ImmutableValidationError: If mutable container detected
    """
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
class ImmutableEvent(RIFTTransformable, RIFTImmutable, RIFTValidatable, RIFTEventProtocol):
    """
    Core immutable event structure following DOP principles
    
    All transformations return new instances, ensuring zero side effects.
    This is the canonical event representation across all Pyics versions.
    
    Cost Impact: Dataclass instantiation + validation overhead
    Thread Safety: Guaranteed through immutable design
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
    
    def transform(self, func: Callable[['ImmutableEvent'], T]) -> T:
        """Apply transformation function maintaining immutability"""
        return func(self)
    
    def validate_integrity(self) -> bool:
        """Validate that this instance maintains immutable constraints"""
        try:
            # Attempt to access all fields to ensure they're still immutable
            for field_name in self.__dataclass_fields__:
                field_value = getattr(self, field_name)
                validate_immutable_field(field_value, field_name, "ImmutableEvent")
            return True
        except ImmutableValidationError:
            return False
    
    def validate_constraints(self) -> bool:
        """Validate business constraints"""
        try:
            # Duration must be positive
            if self.duration.total_seconds() <= 0:
                return False
            
            # UID must not be empty
            if not self.uid.strip():
                return False
            
            # Summary should have reasonable length
            if len(self.summary) > 1000:
                return False
            
            return True
        except Exception:
            return False
    
    def get_validation_errors(self) -> list[str]:
        """Get detailed validation error messages"""
        errors = []
        
        if self.duration.total_seconds() <= 0:
            errors.append("Event duration must be positive")
        
        if not self.uid.strip():
            errors.append("Event UID cannot be empty")
        
        if len(self.summary) > 1000:
            errors.append("Event summary exceeds maximum length (1000 characters)")
        
        if not self.validate_integrity():
            errors.append("Event structure integrity validation failed")
        
        return errors
    
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
        """Create new instance with additional metadata"""
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
class CalendarData(RIFTTransformable, RIFTImmutable, RIFTValidatable, RIFTCalendarProtocol):
    """
    Immutable calendar data structure
    
    Represents complete calendar state without mutable operations.
    All calendar modifications return new instances.
    
    Cost Impact: Event collection processing overhead
    Thread Safety: Guaranteed through immutable design
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
    
    def transform(self, func: Callable[['CalendarData'], T]) -> T:
        """Apply transformation function to calendar data"""
        return func(self)
    
    def validate_integrity(self) -> bool:
        """Validate calendar data maintains immutable constraints"""
        try:
            # Validate all events maintain purity
            for event in self.events:
                if not event.validate_integrity():
                    return False
            
            # Validate top-level immutability
            for field_name in self.__dataclass_fields__:
                if field_na
