#!/usr/bin/env python3
"""
pyics/core/transforms/base.py
Pure Transformation Library for DOP Calendar System

Stateless transformation utilities operating on immutable structures.
All business logic MUST route through these registered transformations.

Author: OBINexus Engineering Team / Nnamdi Okpala
Phase: 3.1 - Core Foundation Implementation
"""

from typing import Callable
from datetime import timedelta
from ..logic.lambda import register_transform
from ..structures.immutables import ImmutableEvent, CalendarData

@register_transform("shift_event_time", version="core")
def shift_event_time(delta: timedelta) -> Callable[[ImmutableEvent], ImmutableEvent]:
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
            tags=event.tags
        )
    return transform_event

@register_transform("add_event_metadata", version="core")
def add_event_metadata(**metadata) -> Callable[[ImmutableEvent], ImmutableEvent]:
    """Create transformation to add metadata to events"""
    def transform_event(event: ImmutableEvent) -> ImmutableEvent:
        return event.with_metadata(**metadata)
    return transform_event

@register_transform("format_event_ics", version="core")
def format_event_ics(event: ImmutableEvent) -> str:
    """Transform event to ICS format"""
    return f"""BEGIN:VEVENT
UID:{event.uid}
SUMMARY:{event.summary}
DTSTART:{event.start_time.strftime('%Y%m%dT%H%M%SZ')}
DURATION:PT{int(event.duration.total_seconds())}S
DESCRIPTION:{event.description}
END:VEVENT"""

# Export transformation functions
__all__ = ['shift_event_time', 'add_event_metadata', 'format_event_ics']
