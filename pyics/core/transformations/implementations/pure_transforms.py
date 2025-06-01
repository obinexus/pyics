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
