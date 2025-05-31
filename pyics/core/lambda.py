#!/usr/bin/env python3
"""
Pyics Core: Data-Oriented Programming with Lambda Calculus
Function composition, aggregation, and extension without overhead

This module demonstrates the theoretical foundation of Pyics:
- Pure function composition using lambda calculus
- Immutable data transformations
- Zero-overhead extension points for future versions
- Modular aggregation without coupling

Author: OBINexus Engineering Team / Nnamdi Okpala
License: MIT
"""

from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, 
    TypeVar, Generic, Protocol, runtime_checkable
)
from functools import reduce, partial, wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import inspect
from collections.abc import Iterable

# Type system for functional composition
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

Transform = Callable[[T], U]
Predicate = Callable[[T], bool]
Aggregator = Callable[[List[T]], U]
Composer = Callable[..., Any]

@runtime_checkable
class Transformable(Protocol[T]):
    """Protocol for data structures that support pure transformations"""
    def transform(self, func: Transform[T, U]) -> U: ...

@runtime_checkable
class Composable(Protocol):
    """Protocol for composable function objects"""
    def compose(self, other: 'Composable') -> 'Composable': ...

# ===== LAMBDA CALCULUS FOUNDATION =====

class Lambda:
    """
    Lambda calculus implementation for pure function composition
    
    This class provides the mathematical foundation for data-oriented
    programming by implementing function composition, currying, and
    partial application without runtime overhead.
    """
    
    @staticmethod
    def identity(x: T) -> T:
        """Identity function: λx.x"""
        return x
    
    @staticmethod
    def constant(value: T) -> Callable[[Any], T]:
        """Constant function: λx.c"""
        return lambda _: value
    
    @staticmethod
    def compose(*functions: Callable) -> Callable:
        """
        Function composition: (f ∘ g)(x) = f(g(x))
        
        Implements mathematical composition with right-to-left evaluation:
        compose(f, g, h)(x) == f(g(h(x)))
        """
        def _compose_two(f: Callable, g: Callable) -> Callable:
            @wraps(f)
            def composed(*args, **kwargs):
                return f(g(*args, **kwargs))
            return composed
        
        if not functions:
            return Lambda.identity
        if len(functions) == 1:
            return functions[0]
        
        return reduce(_compose_two, functions)
    
    @staticmethod
    def pipe(*functions: Callable) -> Callable:
        """
        Function piping: left-to-right composition
        
        pipe(f, g, h)(x) == h(g(f(x)))
        More intuitive for data transformation pipelines
        """
        return Lambda.compose(*reversed(functions))
    
    @staticmethod
    def curry(func: Callable) -> Callable:
        """
        Currying: Transform f(x, y, z) into f(x)(y)(z)
        
        Enables partial application and function specialization
        """
        sig = inspect.signature(func)
        param_count = len(sig.parameters)
        
        def curried(*args):
            if len(args) >= param_count:
                return func(*args[:param_count])
            return lambda *more_args: curried(*(args + more_args))
        
        return curried
    
    @staticmethod
    def partial_apply(func: Callable, *args, **kwargs) -> Callable:
        """Partial application with lambda calculus semantics"""
        return partial(func, *args, **kwargs)

# ===== DATA-ORIENTED PROGRAMMING STRUCTURES =====

@dataclass(frozen=True)
class ImmutableEvent:
    """
    Immutable event structure following DOP principles
    
    All transformations return new instances, ensuring no side effects
    """
    uid: str
    summary: str
    start_time: datetime
    duration: timedelta
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def transform(self, func: Transform['ImmutableEvent', T]) -> T:
        """Apply transformation function to this event"""
        return func(self)
    
    def with_summary(self, summary: str) -> 'ImmutableEvent':
        """Pure transformation: update summary"""
        return self.__class__(
            uid=self.uid,
            summary=summary,
            start_time=self.start_time,
            duration=self.duration,
            description=self.description,
            metadata=self.metadata
        )
    
    def with_metadata(self, **new_metadata) -> 'ImmutableEvent':
        """Pure transformation: merge metadata"""
        merged_metadata = {**self.metadata, **new_metadata}
        return self.__class__(
            uid=self.uid,
            summary=self.summary,
            start_time=self.start_time,
            duration=self.duration,
            description=self.description,
            metadata=merged_metadata
        )

@dataclass(frozen=True)
class CalendarData:
    """
    Immutable calendar data structure
    
    Represents the complete state of a calendar without mutable operations
    """
    events: Tuple[ImmutableEvent, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "v1"
    
    def transform(self, func: Transform['CalendarData', T]) -> T:
        """Apply transformation function to calendar data"""
        return func(self)
    
    def add_event(self, event: ImmutableEvent) -> 'CalendarData':
        """Pure transformation: add event"""
        return CalendarData(
            events=self.events + (event,),
            metadata=self.metadata,
            version=self.version
        )
    
    def filter_events(self, predicate: Predicate[ImmutableEvent]) -> 'CalendarData':
        """Pure transformation: filter events"""
        filtered_events = tuple(filter(predicate, self.events))
        return CalendarData(
            events=filtered_events,
            metadata=self.metadata,
            version=self.version
        )
    
    def map_events(self, transform_func: Transform[ImmutableEvent, ImmutableEvent]) -> 'CalendarData':
        """Pure transformation: map over events"""
        transformed_events = tuple(map(transform_func, self.events))
        return CalendarData(
            events=transformed_events,
            metadata=self.metadata,
            version=self.version
        )

# ===== FUNCTIONAL TRANSFORMATION LIBRARY =====

class Transformations:
    """
    Pure transformation functions for calendar data
    
    All functions are stateless and side-effect free, enabling
    safe composition and parallel execution
    """
    
    @staticmethod
    def add_timezone_metadata(timezone: str) -> Transform[ImmutableEvent, ImmutableEvent]:
        """Create transformation to add timezone metadata"""
        return lambda event: event.with_metadata(timezone=timezone)
    
    @staticmethod
    def validate_event_duration(min_duration: timedelta) -> Transform[ImmutableEvent, bool]:
        """Create validation predicate for minimum duration"""
        return lambda event: event.duration >= min_duration
    
    @staticmethod
    def shift_event_time(delta: timedelta) -> Transform[ImmutableEvent, ImmutableEvent]:
        """Create transformation to shift event time"""
        def shift_time(event: ImmutableEvent) -> ImmutableEvent:
            return ImmutableEvent(
                uid=event.uid,
                summary=event.summary,
                start_time=event.start_time + delta,
                duration=event.duration,
                description=event.description,
                metadata=event.metadata
            )
        return shift_time
    
    @staticmethod
    def format_as_ics_line(event: ImmutableEvent) -> str:
        """Transform event to ICS format line"""
        return f"BEGIN:VEVENT\nUID:{event.uid}\nSUMMARY:{event.summary}\nEND:VEVENT"
    
    @staticmethod
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

# ===== COMPOSITION ENGINE =====

class CompositionEngine:
    """
    Engine for composing transformations without runtime overhead
    
    Provides extension points for future versions while maintaining
    zero-cost abstractions through compile-time composition
    """
    
    def __init__(self, version: str = "v1"):
        self.version = version
        self._registered_transforms: Dict[str, Callable] = {}
        self._extension_points: Dict[str, List[Callable]] = {}
    
    def register_transform(self, name: str, transform: Callable) -> None:
        """Register a named transformation for reuse"""
        self._registered_transforms[name] = transform
    
    def register_extension(self, extension_point: str, handler: Callable) -> None:
        """Register extension handler for future version compatibility"""
        if extension_point not in self._extension_points:
            self._extension_points[extension_point] = []
        self._extension_points[extension_point].append(handler)
    
    def get_transform(self, name: str) -> Optional[Callable]:
        """Retrieve registered transformation"""
        return self._registered_transforms.get(name)
    
    def create_pipeline(self, *transform_names: str) -> Callable:
        """Create composed pipeline from registered transforms"""
        transforms = [
            self._registered_transforms[name] 
            for name in transform_names 
            if name in self._registered_transforms
        ]
        return Lambda.pipe(*transforms)
    
    def extend_pipeline(self, base_pipeline: Callable, extension_point: str) -> Callable:
        """Extend existing pipeline with registered extensions"""
        extensions = self._extension_points.get(extension_point, [])
        if not extensions:
            return base_pipeline
        
        def extended_pipeline(data):
            # Apply base pipeline
            result = base_pipeline(data)
            
            # Apply extensions in sequence
            for extension in extensions:
                result = extension(result)
            
            return result
        
        return extended_pipeline

# ===== PRACTICAL IMPLEMENTATION EXAMPLE =====

class PyicsCore:
    """
    Core Pyics implementation demonstrating data-oriented programming
    
    This class shows how theoretical concepts translate into practical
    calendar automation with zero-overhead abstractions
    """
    
    def __init__(self, version: str = "v2"):
        self.version = version
        self.engine = CompositionEngine(version)
        self._setup_core_transforms()
    
    def _setup_core_transforms(self) -> None:
        """Register core transformations for composition"""
        # Basic transformations
        self.engine.register_transform(
            "add_timezone", 
            Transformations.add_timezone_metadata("UTC")
        )
        
        self.engine.register_transform(
            "validate_duration",
            Transformations.validate_event_duration(timedelta(minutes=15))
        )
        
        self.engine.register_transform(
            "format_ics",
            Transformations.format_as_ics_line
        )
        
        # Extension points for future versions
        self.engine.register_extension(
            "pre_processing",
            lambda data: data.with_metadata(processed_by=self.version)
        )
        
        if self.version >= "v2":
            self.engine.register_extension(
                "post_processing",
                lambda data: data.with_metadata(enhanced=True)
            )
    
    def create_milestone_calendar(
        self,
        start_date: datetime,
        milestones: List[str],
        interval_days: int = 14
    ) -> CalendarData:
        """
        Create milestone calendar using pure functional composition
        
        Demonstrates data-oriented programming with immutable transformations
        """
        # Create base events using pure functions
        events = []
        for i, milestone in enumerate(milestones):
            event_time = start_date + timedelta(days=i * interval_days)
            event = ImmutableEvent(
                uid=f"milestone-{i+1}",
                summary=milestone,
                start_time=event_time,
                duration=timedelta(hours=1),
                description=f"Milestone: {milestone}"
            )
            events.append(event)
        
        # Create calendar data
        calendar_data = CalendarData(events=tuple(events))
        
        # Apply transformations through composition
        timezone_transform = self.engine.get_transform("add_timezone")
        if timezone_transform:
            calendar_data = calendar_data.map_events(timezone_transform)
        
        # Apply extensions for version compatibility
        extended_pipeline = self.engine.extend_pipeline(
            Lambda.identity,
            "pre_processing"
        )
        calendar_data = extended_pipeline(calendar_data)
        
        return calendar_data
    
    def generate_ics_content(self, calendar_data: CalendarData) -> str:
        """
        Generate ICS content using functional composition
        
        Pure transformation from calendar data to ICS format
        """
        # Header
        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "CALSCALE:GREGORIAN",
            f"PRODID:-//Pyics {self.version}//EN"
        ]
        
        # Transform events to ICS format
        format_transform = self.engine.get_transform("format_ics")
        if format_transform:
            event_lines = [
                format_transform(event) 
                for event in calendar_data.events
            ]
            ics_lines.extend(event_lines)
        
        # Footer
        ics_lines.append("END:VCALENDAR")
        
        return "\n".join(ics_lines)
    
    def compose_custom_pipeline(self, *operations: str) -> Callable:
        """
        Create custom transformation pipeline
        
        Enables users to compose their own processing chains
        """
        return self.engine.create_pipeline(*operations)

# ===== USAGE DEMONSTRATION =====

def demonstrate_dop_approach():
    """
    Demonstrate data-oriented programming approach with Pyics
    
    Shows how lambda calculus and pure functions enable
    zero-overhead extensibility
    """
    print("=== Pyics Data-Oriented Programming Demo ===")
    
    # Initialize core with version support
    core_v1 = PyicsCore(version="v1")
    core_v2 = PyicsCore(version="v2")
    
    # Create milestone calendar using pure functions
    start_date = datetime(2024, 12, 30, 9, 0)
    milestones = [
        "Initial Violation (£1M)",
        "Continued Breach (£1M)",
        "Systemic Neglect (£1M)"
    ]
    
    # Generate calendar with v1 (basic processing)
    calendar_v1 = core_v1.create_milestone_calendar(start_date, milestones)
    print(f"V1 Events: {len(calendar_v1.events)}")
    print(f"V1 Metadata: {calendar_v1.metadata}")
    
    # Generate calendar with v2 (enhanced processing)
    calendar_v2 = core_v2.create_milestone_calendar(start_date, milestones)
    print(f"V2 Events: {len(calendar_v2.events)}")
    print(f"V2 Metadata: {calendar_v2.metadata}")
    
    # Demonstrate function composition
    shift_and_validate = Lambda.compose(
        Transformations.validate_event_duration(timedelta(minutes=30)),
        Transformations.shift_event_time(timedelta(hours=1))
    )
    
    # Apply composed transformation
    first_event = calendar_v2.events[0]
    shifted_event = Transformations.shift_event_time(timedelta(hours=1))(first_event)
    print(f"Original time: {first_event.start_time}")
    print(f"Shifted time: {shifted_event.start_time}")
    
    # Generate ICS content
    ics_content = core_v2.generate_ics_content(calendar_v2)
    print(f"ICS Content Length: {len(ics_content)} characters")
    
    print("=== Demo Complete ===")

if __name__ == "__main__":
    demonstrate_dop_approach()

