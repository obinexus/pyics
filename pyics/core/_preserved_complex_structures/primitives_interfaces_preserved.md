# Preserved Complex Structure: primitives/interfaces

Preservation timestamp: 2025-06-01T23:50:03.438064
Systematic cleanup phase: structure_flattening

## data_types.py
```python
#!/usr/bin/env python3
"""
pyics/core/primitives/data_types.py
Pyics Core Domain Data Types: primitives

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: primitives
Responsibility: Core data containers and type definitions
Compute Weight: Static (immutable data structures)

PROBLEM SOLVED: Centralized type definitions for primitives domain
DEPENDENCIES: Standard library typing, dataclasses
THREAD SAFETY: Yes - immutable data structures
DETERMINISTIC: Yes - static type definitions

This module defines the core data types and structures for the primitives
domain following Data-Oriented Programming principles with immutable,
composable data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Union, Protocol
from enum import Enum, auto
from datetime import datetime

# Domain-specific enums
class PrimitivesStatus(Enum):
    """Status enumeration for primitives domain operations"""
    INITIALIZED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    ERROR = auto()

class PrimitivesPriority(Enum):
    """Priority levels for primitives domain elements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Core data containers
@dataclass(frozen=True)
class PrimitivesEntity:
    """
    Base entity for primitives domain
    
    Immutable data container following DOP principles
    """
    id: str
    name: str
    status: PrimitivesStatus = PrimitivesStatus.INITIALIZED
    priority: PrimitivesPriority = PrimitivesPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PrimitivesConfig:
    """
    Configuration data structure for primitives domain
    
    Static configuration with validation support
    """
    enabled: bool = True
    max_items: int = 1000
    timeout_seconds: int = 30
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PrimitivesResult:
    """
    Result container for primitives domain operations
    
    Immutable result with success/error handling
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Protocol definitions for type checking
class PrimitivesProcessor(Protocol):
    """Protocol for primitives domain processors"""
    
    def process(self, entity: PrimitivesEntity) -> PrimitivesResult:
        """Process a primitives entity"""
        ...
    
    def validate(self, entity: PrimitivesEntity) -> bool:
        """Validate a primitives entity"""
        ...

class PrimitivesRepository(Protocol):
    """Protocol for primitives domain data repositories"""
    
    def store(self, entity: PrimitivesEntity) -> bool:
        """Store a primitives entity"""
        ...
    
    def retrieve(self, entity_id: str) -> Optional[PrimitivesEntity]:
        """Retrieve a primitives entity by ID"""
        ...
    
    def list_all(self) -> List[PrimitivesEntity]:
        """List all primitives entities"""
        ...

# Type aliases for complex structures
PrimitivesCollection = List[PrimitivesEntity]
PrimitivesIndex = Dict[str, PrimitivesEntity]
PrimitivesFilter = Dict[str, Any]

# Export interface
__all__ = [
    'PrimitivesStatus',
    'PrimitivesPriority',
    'PrimitivesEntity',
    'PrimitivesConfig',
    'PrimitivesResult',
    'PrimitivesProcessor',
    'PrimitivesRepository',
    'PrimitivesCollection',
    'PrimitivesIndex',
    'PrimitivesFilter',
]

# [EOF] - End of primitives data_types.py module

```

## operations.py
```python
#!/usr/bin/env python3
"""
pyics/core/primitives/operations.py
Pyics Core Domain Operations: primitives

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: primitives
Responsibility: Atomic and composed operations on domain data
Compute Weight: Dynamic (varies by operation complexity)

PROBLEM SOLVED: Centralized operation definitions for primitives domain
DEPENDENCIES: primitives.data_types, primitives.relations, typing
THREAD SAFETY: Yes - pure functions with immutable data
DETERMINISTIC: Yes - deterministic operations on immutable data

This module provides atomic and composed operations for the primitives
domain, implementing pure functions that transform immutable data structures
following DOP principles.
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator, Any
from functools import reduce, partial
from dataclasses import replace
import logging

# Import domain data types and relations
from .data_types import (
    PrimitivesEntity,
    PrimitivesConfig,
    PrimitivesResult,
    PrimitivesCollection,
    PrimitivesIndex,
    PrimitivesFilter,
    PrimitivesStatus,
    PrimitivesPriority
)
from .relations import RelationGraph, Relation, RelationType

logger = logging.getLogger(f"pyics.core.primitives.operations")

# Atomic operations (pure functions)
def create_entity(
    entity_id: str,
    name: str,
    status: PrimitivesStatus = PrimitivesStatus.INITIALIZED,
    priority: PrimitivesPriority = PrimitivesPriority.MEDIUM,
    **metadata
) -> PrimitivesEntity:
    """
    Create a new primitives entity
    
    Pure function for entity creation
    """
    return PrimitivesEntity(
        id=entity_id,
        name=name,
        status=status,
        priority=priority,
        metadata=metadata
    )

def update_entity_status(
    entity: PrimitivesEntity,
    new_status: PrimitivesStatus
) -> PrimitivesEntity:
    """
    Update entity status (returns new entity)
    
    Pure function for status updates
    """
    return replace(entity, status=new_status)

def update_entity_priority(
    entity: PrimitivesEntity,
    new_priority: PrimitivesPriority
) -> PrimitivesEntity:
    """
    Update entity priority (returns new entity)
    
    Pure function for priority updates
    """
    return replace(entity, priority=new_priority)

def add_entity_metadata(
    entity: PrimitivesEntity,
    key: str,
    value: Any
) -> PrimitivesEntity:
    """
    Add metadata to entity (returns new entity)
    
    Pure function for metadata updates
    """
    new_metadata = {**entity.metadata, key: value}
    return replace(entity, metadata=new_metadata)

# Collection operations (pure functions)
def filter_entities_by_status(
    entities: PrimitivesCollection,
    status: PrimitivesStatus
) -> PrimitivesCollection:
    """
    Filter entities by status
    
    Pure filtering function
    """
    return [entity for entity in entities if entity.status == status]

def filter_entities_by_priority(
    entities: PrimitivesCollection,
    min_priority: PrimitivesPriority
) -> PrimitivesCollection:
    """
    Filter entities by minimum priority
    
    Pure filtering function
    """
    return [
        entity for entity in entities 
        if entity.priority.value >= min_priority.value
    ]

def sort_entities_by_priority(
    entities: PrimitivesCollection,
    descending: bool = True
) -> PrimitivesCollection:
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
    entities: PrimitivesCollection
) -> Dict[PrimitivesStatus, PrimitivesCollection]:
    """
    Group entities by status
    
    Pure grouping function
    """
    groups: Dict[PrimitivesStatus, PrimitivesCollection] = {}
    
    for entity in entities:
        if entity.status not in groups:
            groups[entity.status] = []
        groups[entity.status].append(entity)
    
    return groups

# Index operations (pure functions)
def build_entity_index(
    entities: PrimitivesCollection
) -> PrimitivesIndex:
    """
    Build index from entity collection
    
    Pure function for index creation
    """
    return {entity.id: entity for entity in entities}

def merge_entity_indices(
    *indices: PrimitivesIndex
) -> PrimitivesIndex:
    """
    Merge multiple entity indices
    
    Pure function for index merging
    """
    merged = {}
    for index in indices:
        merged.update(index)
    return merged

def filter_index_by_predicate(
    index: PrimitivesIndex,
    predicate: Callable[[PrimitivesEntity], bool]
) -> PrimitivesIndex:
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
    entities: PrimitivesCollection,
    operations: List[Callable[[PrimitivesEntity], PrimitivesEntity]]
) -> PrimitivesCollection:
    """
    Apply a sequence of operations to entity collection
    
    Composed operation using function composition
    """
    def apply_operations(entity: PrimitivesEntity) -> PrimitivesEntity:
        return reduce(lambda e, op: op(e), operations, entity)
    
    return [apply_operations(entity) for entity in entities]

def transform_collection_with_config(
    entities: PrimitivesCollection,
    config: PrimitivesConfig
) -> PrimitivesResult:
    """
    Transform collection based on configuration
    
    Composed operation with result wrapping
    """
    try:
        # Apply configuration-based transformations
        filtered_entities = entities[:config.max_items] if config.max_items > 0 else entities
        
        if not config.enabled:
            return PrimitivesResult(
                success=True,
                data=filtered_entities,
                metadata={"config_enabled": False}
            )
        
        # Process entities based on configuration
        processed_entities = []
        for entity in filtered_entities:
            # Apply configuration-specific processing
            if config.options.get("auto_priority_boost", False):
                entity = update_entity_priority(entity, PrimitivesPriority.HIGH)
            
            processed_entities.append(entity)
        
        return PrimitivesResult(
            success=True,
            data=processed_entities,
            metadata={
                "processed_count": len(processed_entities),
                "config_applied": True
            }
        )
    
    except Exception as e:
        logger.error(f"Collection transformation failed: {e}")
        return PrimitivesResult(
            success=False,
            error_message=str(e),
            metadata={"operation": "transform_collection_with_config"}
        )

def validate_entity_collection(
    entities: PrimitivesCollection,
    validation_rules: List[Callable[[PrimitivesEntity], bool]]
) -> PrimitivesResult:
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
    
    return PrimitivesResult(
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
) -> PrimitivesCollection:
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
    partial_operation(update_entity_status, new_status=PrimitivesStatus.PROCESSING),
]

PRIORITY_BOOST_OPERATIONS = [
    partial_operation(update_entity_priority, new_priority=PrimitivesPriority.HIGH),
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

# [EOF] - End of primitives operations.py module

```

## relations.py
```python
#!/usr/bin/env python3
"""
pyics/core/primitives/relations.py
Pyics Core Domain Relations: primitives

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: primitives
Responsibility: Structural mappings and relationship definitions
Compute Weight: Static to Computed (depending on relation complexity)

PROBLEM SOLVED: Centralized relationship mapping for primitives domain
DEPENDENCIES: primitives.data_types, typing, dataclasses
THREAD SAFETY: Yes - immutable relation structures
DETERMINISTIC: Yes - static relationship definitions

This module defines structural relationships and mappings between entities
in the primitives domain, following DOP principles with immutable
relation containers and pure transformation functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator
from enum import Enum, auto

# Import domain data types
from .data_types import (
    PrimitivesEntity,
    PrimitivesCollection,
    PrimitivesIndex,
    PrimitivesFilter
)

# Relationship types
class RelationType(Enum):
    """Types of relationships in primitives domain"""
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
    Immutable relation between primitives entities
    
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
    Immutable graph of relations in primitives domain
    
    Container for complete relationship structure
    """
    relations: Tuple[Relation, ...] = field(default_factory=tuple)
    entity_index: Dict[str, PrimitivesEntity] = field(default_factory=dict)
    
    def get_relations_for_entity(self, entity_id: str) -> List[Relation]:
        """Get all relations involving an entity"""
        return [
            rel for rel in self.relations 
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
    
    def get_related_entities(self, entity_id: str) -> List[PrimitivesEntity]:
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
    Static mapping configuration for primitives relations
    
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
    entities: PrimitivesCollection,
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

# Predefined relation mappings for primitives domain
DEFAULT_RELATION_MAPPINGS: List[RelationMapping] = [
    RelationMapping(
        mapping_name="hierarchical_parent_child",
        source_type="PrimitivesEntity",
        target_type="PrimitivesEntity",
        relation_type=RelationType.HIERARCHICAL,
        validation_rules=("source_id != target_id", "no_circular_dependencies")
    ),
    RelationMapping(
        mapping_name="dependency_chain",
        source_type="PrimitivesEntity",
        target_type="PrimitivesEntity",
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

# [EOF] - End of primitives relations.py module

```

## __init__.py
```python

```

