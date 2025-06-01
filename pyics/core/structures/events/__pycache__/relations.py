#!/usr/bin/env python3
"""
pyics/core/structures/events/relations.py
Pyics Core Domain Relations: structures/events

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: structures/events
Responsibility: Structural mappings and relationship definitions
Compute Weight: Static to Computed (depending on relation complexity)

PROBLEM SOLVED: Centralized relationship mapping for structures/events domain
DEPENDENCIES: structures/events.data_types, typing, dataclasses
THREAD SAFETY: Yes - immutable relation structures
DETERMINISTIC: Yes - static relationship definitions

This module defines structural relationships and mappings between entities
in the structures/events domain, following DOP principles with immutable
relation containers and pure transformation functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator
from enum import Enum, auto

# Import domain data types
from .data_types import (
    Structures/EventsEntity,
    Structures/EventsCollection,
    Structures/EventsIndex,
    Structures/EventsFilter
)

# Relationship types
class RelationType(Enum):
    """Types of relationships in structures/events domain"""
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
    Immutable relation between structures/events entities
    
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
    Immutable graph of relations in structures/events domain
    
    Container for complete relationship structure
    """
    relations: Tuple[Relation, ...] = field(default_factory=tuple)
    entity_index: Dict[str, Structures/EventsEntity] = field(default_factory=dict)
    
    def get_relations_for_entity(self, entity_id: str) -> List[Relation]:
        """Get all relations involving an entity"""
        return [
            rel for rel in self.relations 
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
    
    def get_related_entities(self, entity_id: str) -> List[Structures/EventsEntity]:
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
    Static mapping configuration for structures/events relations
    
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
    entities: Structures/EventsCollection,
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

# Predefined relation mappings for structures/events domain
DEFAULT_RELATION_MAPPINGS: List[RelationMapping] = [
    RelationMapping(
        mapping_name="hierarchical_parent_child",
        source_type="Structures/EventsEntity",
        target_type="Structures/EventsEntity",
        relation_type=RelationType.HIERARCHICAL,
        validation_rules=("source_id != target_id", "no_circular_dependencies")
    ),
    RelationMapping(
        mapping_name="dependency_chain",
        source_type="Structures/EventsEntity",
        target_type="Structures/EventsEntity",
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

# [EOF] - End of structures/events relations.py module
