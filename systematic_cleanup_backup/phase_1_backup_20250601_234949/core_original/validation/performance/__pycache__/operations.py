#!/usr/bin/env python3
"""
pyics/core/validation/performance/operations.py
Pyics Core Domain Operations: validation/performance

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: validation/performance
Responsibility: Atomic and composed operations on domain data
Compute Weight: Dynamic (varies by operation complexity)

PROBLEM SOLVED: Centralized operation definitions for validation/performance domain
DEPENDENCIES: validation/performance.data_types, validation/performance.relations, typing
THREAD SAFETY: Yes - pure functions with immutable data
DETERMINISTIC: Yes - deterministic operations on immutable data

This module provides atomic and composed operations for the validation/performance
domain, implementing pure functions that transform immutable data structures
following DOP principles.
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator, Any
from functools import reduce, partial
from dataclasses import replace
import logging

# Import domain data types and relations
from .data_types import (
    Validation/PerformanceEntity,
    Validation/PerformanceConfig,
    Validation/PerformanceResult,
    Validation/PerformanceCollection,
    Validation/PerformanceIndex,
    Validation/PerformanceFilter,
    Validation/PerformanceStatus,
    Validation/PerformancePriority
)
from .relations import RelationGraph, Relation, RelationType

logger = logging.getLogger(f"pyics.core.validation/performance.operations")

# Atomic operations (pure functions)
def create_entity(
    entity_id: str,
    name: str,
    status: Validation/PerformanceStatus = Validation/PerformanceStatus.INITIALIZED,
    priority: Validation/PerformancePriority = Validation/PerformancePriority.MEDIUM,
    **metadata
) -> Validation/PerformanceEntity:
    """
    Create a new validation/performance entity
    
    Pure function for entity creation
    """
    return Validation/PerformanceEntity(
        id=entity_id,
        name=name,
        status=status,
        priority=priority,
        metadata=metadata
    )

def update_entity_status(
    entity: Validation/PerformanceEntity,
    new_status: Validation/PerformanceStatus
) -> Validation/PerformanceEntity:
    """
    Update entity status (returns new entity)
    
    Pure function for status updates
    """
    return replace(entity, status=new_status)

def update_entity_priority(
    entity: Validation/PerformanceEntity,
    new_priority: Validation/PerformancePriority
) -> Validation/PerformanceEntity:
    """
    Update entity priority (returns new entity)
    
    Pure function for priority updates
    """
    return replace(entity, priority=new_priority)

def add_entity_metadata(
    entity: Validation/PerformanceEntity,
    key: str,
    value: Any
) -> Validation/PerformanceEntity:
    """
    Add metadata to entity (returns new entity)
    
    Pure function for metadata updates
    """
    new_metadata = {**entity.metadata, key: value}
    return replace(entity, metadata=new_metadata)

# Collection operations (pure functions)
def filter_entities_by_status(
    entities: Validation/PerformanceCollection,
    status: Validation/PerformanceStatus
) -> Validation/PerformanceCollection:
    """
    Filter entities by status
    
    Pure filtering function
    """
    return [entity for entity in entities if entity.status == status]

def filter_entities_by_priority(
    entities: Validation/PerformanceCollection,
    min_priority: Validation/PerformancePriority
) -> Validation/PerformanceCollection:
    """
    Filter entities by minimum priority
    
    Pure filtering function
    """
    return [
        entity for entity in entities 
        if entity.priority.value >= min_priority.value
    ]

def sort_entities_by_priority(
    entities: Validation/PerformanceCollection,
    descending: bool = True
) -> Validation/PerformanceCollection:
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
    entities: Validation/PerformanceCollection
) -> Dict[Validation/PerformanceStatus, Validation/PerformanceCollection]:
    """
    Group entities by status
    
    Pure grouping function
    """
    groups: Dict[Validation/PerformanceStatus, Validation/PerformanceCollection] = {}
    
    for entity in entities:
        if entity.status not in groups:
            groups[entity.status] = []
        groups[entity.status].append(entity)
    
    return groups

# Index operations (pure functions)
def build_entity_index(
    entities: Validation/PerformanceCollection
) -> Validation/PerformanceIndex:
    """
    Build index from entity collection
    
    Pure function for index creation
    """
    return {entity.id: entity for entity in entities}

def merge_entity_indices(
    *indices: Validation/PerformanceIndex
) -> Validation/PerformanceIndex:
    """
    Merge multiple entity indices
    
    Pure function for index merging
    """
    merged = {}
    for index in indices:
        merged.update(index)
    return merged

def filter_index_by_predicate(
    index: Validation/PerformanceIndex,
    predicate: Callable[[Validation/PerformanceEntity], bool]
) -> Validation/PerformanceIndex:
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
    entities: Validation/PerformanceCollection,
    operations: List[Callable[[Validation/PerformanceEntity], Validation/PerformanceEntity]]
) -> Validation/PerformanceCollection:
    """
    Apply a sequence of operations to entity collection
    
    Composed operation using function composition
    """
    def apply_operations(entity: Validation/PerformanceEntity) -> Validation/PerformanceEntity:
        return reduce(lambda e, op: op(e), operations, entity)
    
    return [apply_operations(entity) for entity in entities]

def transform_collection_with_config(
    entities: Validation/PerformanceCollection,
    config: Validation/PerformanceConfig
) -> Validation/PerformanceResult:
    """
    Transform collection based on configuration
    
    Composed operation with result wrapping
    """
    try:
        # Apply configuration-based transformations
        filtered_entities = entities[:config.max_items] if config.max_items > 0 else entities
        
        if not config.enabled:
            return Validation/PerformanceResult(
                success=True,
                data=filtered_entities,
                metadata={"config_enabled": False}
            )
        
        # Process entities based on configuration
        processed_entities = []
        for entity in filtered_entities:
            # Apply configuration-specific processing
            if config.options.get("auto_priority_boost", False):
                entity = update_entity_priority(entity, Validation/PerformancePriority.HIGH)
            
            processed_entities.append(entity)
        
        return Validation/PerformanceResult(
            success=True,
            data=processed_entities,
            metadata={
                "processed_count": len(processed_entities),
                "config_applied": True
            }
        )
    
    except Exception as e:
        logger.error(f"Collection transformation failed: {e}")
        return Validation/PerformanceResult(
            success=False,
            error_message=str(e),
            metadata={"operation": "transform_collection_with_config"}
        )

def validate_entity_collection(
    entities: Validation/PerformanceCollection,
    validation_rules: List[Callable[[Validation/PerformanceEntity], bool]]
) -> Validation/PerformanceResult:
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
    
    return Validation/PerformanceResult(
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
) -> Validation/PerformanceCollection:
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
    partial_operation(update_entity_status, new_status=Validation/PerformanceStatus.PROCESSING),
]

PRIORITY_BOOST_OPERATIONS = [
    partial_operation(update_entity_priority, new_priority=Validation/PerformancePriority.HIGH),
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

# [EOF] - End of validation/performance operations.py module
