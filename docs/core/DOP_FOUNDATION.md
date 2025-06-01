# Pyics DOP Foundation Architecture

## Overview

The Data-Oriented Programming (DOP) foundation provides mathematical rigor and functional correctness for the Pyics calendar automation system. This foundation enforces zero-trust principles through pure function composition and immutable data structures.

## Core Components

### 1. Lambda Calculus Engine (`logic/lambda.py`)
- **compose**: Right-to-left function composition
- **pipe**: Left-to-right function composition  
- **curry**: Partial application support
- **identity**: Mathematical identity function

### 2. Immutable Structures (`structures/immutables.py`)
- **ImmutableEvent**: Frozen dataclass for calendar events
- **CalendarData**: Immutable calendar container
- **Enumerations**: Type-safe status and priority definitions

### 3. Pure Transformations (`transforms/base.py`)
- **shift_event_time**: Time manipulation without mutation
- **add_event_metadata**: Metadata merging with immutability
- **format_event_ics**: ICS format generation

### 4. Validation Framework (`validation/purity.py`)
- **Function purity validation**: Side-effect detection
- **Immutability enforcement**: Return value validation
- **Composition correctness**: Mathematical verification

## Usage Guidelines

### Transformation Registration
```python
from pyics.core import register_transform

@register_transform("my_transform", version="v1")
def my_pure_function(data):
    return transformed_data
```

### Function Composition
```python
from pyics.core import Lambda

pipeline = Lambda.pipe(
    transform_one,
    transform_two,
    transform_three
)
```

### Immutable Operations
```python
from pyics.core import ImmutableEvent

event = ImmutableEvent(...)
modified_event = event.with_metadata(location="Room A")
# Original event remains unchanged
```

## Compliance Requirements

1. **Zero Trust**: All operations must route through registered transforms
2. **Purity**: Functions must not mutate state or produce side effects  
3. **Immutability**: Data structures must remain frozen after creation
4. **Composition**: Complex operations must use lambda calculus primitives

## Integration with Version Modules

All version-specific modules (v1/, v2/, v3-preview/) MUST:
- Import transformations from core registry
- Use immutable structures for state management
- Route all operations through composition engine
- Validate purity before registration

## Testing Strategy

- **Mathematical correctness**: Verify composition laws
- **Immutability preservation**: Ensure no state mutation
- **Performance validation**: Benchmark composition overhead
- **Integration testing**: Validate cross-version compatibility
