#!/usr/bin/env python3
"""
pyics/core/transforms/pipeline/operations.py
Pyics Core Domain Operations: transforms/pipeline

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: transforms/pipeline
Responsibility: Atomic and composed operations on domain data
Compute Weight: Dynamic (varies by operation complexity)

PROBLEM SOLVED: Centralized operation definitions for transforms/pipeline domain
DEPENDENCIES: transforms/pipeline.data_types, transforms/pipeline.relations, typing
THREAD SAFETY: Yes - pure functions with immutable data
DETERMINISTIC: Yes - deterministic operations on immutable data

This module provides atomic and composed operations for the transforms/pipeline
domain, implementing pure functions that transform immutable data structures
following DOP principles.
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator, Any
from functools import reduce, partial
from dataclasses import replace
import logging

# Import domain data types and relations
from .data_types import (
    Transforms/PipelineEntity,
    Transforms/PipelineConfig,
    Transforms/PipelineResult,
    Transforms/PipelineCollection,
    Transforms/PipelineIndex,
    Transforms/PipelineFilter,
    Transforms/PipelineStatus,
    Transforms/PipelinePriority
)
from .relations import RelationGraph, Relation, RelationType

logger = logging.getLogger(f"pyics.core.transforms/pipeline.operations")

# Atomic operations (pure functions)
def create_entity(
    entity_id: str,
    name: str,
    status: Transforms/PipelineStatus = Transforms/PipelineStatus.INITIALIZED,
    priority: Transforms/PipelinePriority = Transforms/PipelinePriority.MEDIUM,
    **metadata
) -> Transforms/PipelineEntity:
    """
    Create a new transforms/pipeline entity
    
    Pure function for entity creation
    """
    return Transforms/PipelineEntity(
        id=entity_id,
        name=name,
        status=status,
        priority=priority,
        metadata=metadata
    )

def update_entity_status(
    entity: Transforms/PipelineEntity,
    new_status: Transforms/PipelineStatus
) -> Transforms/PipelineEntity:
    """
    Update entity status (returns new entity)
    
    Pure function for status updates
    """
    return replace(entity, status=new_status)

def update_entity_priority(
    entity: Transforms/PipelineEntity,
    new_priority: Transforms/PipelinePriority
) -> Transforms/PipelineEntity:
    """
    Update entity priority (returns new entity)
    
    Pure function for priority updates
    """
    return replace(entity, priority=new_priority)

def add_entity_metadata(
    entity: Transforms/PipelineEntity,
    key: str,
    value: Any
) -> Transforms/PipelineEntity:
    """
    Add metadata to entity (returns new entity)
    
    Pure function for metadata updates
    """
    new_metadata = {**entity.metadata, key: value}
    return replace(entity, metadata=new_metadata)

# Collection operations (pure functions)
def filter_entities_by_status(
    entities: Transforms/PipelineCollection,
    status: Transforms/PipelineStatus
) -> Transforms/PipelineCollection:
    """
    Filter entities by status
    
    Pure filtering function
    """
    return [entity for entity in entities if entity.status == status]

def filter_entities_by_priority(
    entities: Transforms/PipelineCollection,
    min_priority: Transforms/PipelinePriority
) -> Transforms/PipelineCollection:
    """
    Filter entities by minimum priority
    
    Pure filtering function
    """
    return [
        entity for entity in entities 
        if entity.priority.value >= min_priority.value
    ]

def sort_entities_by_priority(
    entities: Transforms/PipelineCollection,
    descending: bool = True
) -> Transforms/PipelineCollection:
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
    entities: Transforms/PipelineCollection
) -> Dict[Transforms/PipelineStatus, Transforms/PipelineCollection]:
    """
    Group entities by status
    
    Pure grouping function
    """
    groups: Dict[Transforms/PipelineStatus, Transforms/PipelineCollection] = {}
    
    for entity in entities:
        if entity.status not in groups:
            groups[entity.status] = []
        groups[entity.status].append(entity)
    
    return groups

# Index operations (pure functions)
def build_entity_index(
    entities: Transforms/PipelineCollection
) -> Transforms/PipelineIndex:
    """
    Build index from entity collection
    
    Pure function for index creation
    """
    return {entity.id: entity for entity in entities}

def merge_entity_indices(
    *indices: Transforms/PipelineIndex
) -> Transforms/PipelineIndex:
    """
    Merge multiple entity indices
    
    Pure function for index merging
    """
    merged = {}
    for index in indices:
        merged.update(index)
    return merged

def filter_index_by_predicate(
    index: Transforms/PipelineIndex,
    predicate: Callable[[Transforms/PipelineEntity], bool]
) -> Transforms/PipelineIndex:
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
    entities: Transforms/PipelineCollection,
    operations: List[Callable[[Transforms/PipelineEntity], Transforms/PipelineEntity]]
) -> Transforms/PipelineCollection:
    """
    Apply a sequence of operations to entity collection
    
    Composed operation using function composition
    """
    def apply_operations(entity: Transforms/PipelineEntity) -> Transforms/PipelineEntity:
        return reduce(lambda e, op: op(e), operations, entity)
    
    return [apply_operations(entity) for entity in entities]

def transform_collection_with_config(
    entities: Transforms/PipelineCollection,
    config: Transforms/PipelineConfig
) -> Transforms/PipelineResult:
    """
    Transform collection based on configuration
    
    Composed operation with result wrapping
    """
    try:
        # Apply configuration-based transformations
        filtered_entities = entities[:config.max_items] if config.max_items > 0 else entities
        
        if not config.enabled:
            return Transforms/PipelineResult(
                success=True,
                data=filtered_entities,
                metadata={"config_enabled": False}
            )
        
        # Process entities based on configuration
        processed_entities = []
        for entity in filtered_entities:
            # Apply configuration-specific processing
            if config.options.get("auto_priority_boost", False):
                entity = update_entity_priority(entity, Transforms/PipelinePriority.HIGH)
            
            processed_entities.append(entity)
        
        return Transforms/PipelineResult(
            success=True,
            data=processed_entities,
            metadata={
                "processed_count": len(processed_entities),
                "config_applied": True
            }
        )
    
    except Exception as e:
        logger.error(f"Collection transformation failed: {e}")
        return Transforms/PipelineResult(
            success=False,
            error_message=str(e),
            metadata={"operation": "transform_collection_with_config"}
        )

def validate_entity_collection(
    entities: Transforms/PipelineCollection,
    validation_rules: List[Callable[[Transforms/PipelineEntity], bool]]
) -> Transforms/PipelineResult:
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
    
    return Transforms/PipelineResult(
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
) -> Transforms/PipelineCollection:
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
    partial_operation(update_entity_status, new_status=Transforms/PipelineStatus.PROCESSING),
]

PRIORITY_BOOST_OPERATIONS = [
    partial_operation(update_entity_priority, new_priority=Transforms/PipelinePriority.HIGH),
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

# [EOF] - End of transforms/pipeline operations.py module
