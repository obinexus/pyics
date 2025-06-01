#!/usr/bin/env python3
"""
pyics/core/structures/immutables.py
Immutable Data Structures for DOP Calendar System

All Pyics data flows through these immutable structures.
Zero Trust Principle: No direct state mutation allowed.

Author: OBINexus Engineering Team / Nnamdi Okpala
Phase: 3.1 - Core Foundation Implementation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, FrozenSet
from datetime import datetime, timedelta
from enum import Enum

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

@dataclass(frozen=True)
class ImmutableEvent:
    """Core immutable event structure following DOP principles"""
    uid: str
    summary: str
    start_time: datetime
    duration: timedelta
    description: str = ""
    status: EventStatus = EventStatus.DRAFT
    priority: PriorityLevel = PriorityLevel.MEDIUM
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    tags: FrozenSet[str] = field(default_factory=frozenset)
    
    def with_metadata(self, **new_metadata) -> 'ImmutableEvent':
        """Pure transformation: merge metadata"""
        existing_meta = dict(self.metadata)
        merged_metadata = {**existing_meta, **new_metadata}
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
            tags=self.tags
        )

@dataclass(frozen=True)
class CalendarData:
    """Immutable calendar data structure"""
    events: Tuple[ImmutableEvent, ...] = field(default_factory=tuple)
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    version: str = "v1"
    name: str = ""
    
    def add_event(self, event: ImmutableEvent) -> 'CalendarData':
        """Pure transformation: add event"""
        return CalendarData(
            events=self.events + (event,),
            metadata=self.metadata,
            version=self.version,
            name=self.name
        )

# Export core structures
__all__ = ['ImmutableEvent', 'CalendarData', 'EventStatus', 'PriorityLevel']
