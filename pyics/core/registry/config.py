#!/usr/bin/env python3
"""
pyics/core/registry/config.py
Domain Configuration Module - Auto-Generated

Generated: 2025-05-31T19:27:17.694715
Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: registry

PROBLEM SOLVED: Provides centralized configuration and cost metadata for registry domain
DEPENDENCIES: pyics.core.registry components
THREAD SAFETY: Yes - immutable configuration data
DETERMINISTIC: Yes - static configuration with predictable behavior

This module provides DOP-compliant configuration management for the registry domain,
including cost metadata, behavior policies, and dependency injection interfaces.
"""

from typing import Dict, List, Any, TypedDict, Literal, Optional
import logging

logger = logging.getLogger(f"pyics.core.registry.config")

# Type definitions for configuration
class DomainCostMetadata(TypedDict):
    priority_index: int
    compute_time_weight: float
    exposure_type: Literal["public", "internal", "private"]
    dependency_level: int
    thread_safe: bool
    load_order: int

class DomainConfiguration(TypedDict):
    domain_name: str
    cost_metadata: DomainCostMetadata
    data_types_available: List[str]
    relations_defined: List[str]
    behavior_policies: Dict[str, Any]
    export_interface: List[str]

# Cost metadata for registry domain
cost_metadata: DomainCostMetadata = {
    "priority_index": 90,
    "compute_time_weight": 2.2,
    "exposure_type": "private",
    "dependency_level": 3,
    "thread_safe": True,
    "load_order": 90
}

# Data types available in this domain
DATA_TYPES_AVAILABLE: List[str] = ['CoreStatus', 'CorePriority', 'CoreEntity', 'CoreConfig', 'CoreResult', 'CoreProcessor', 'CoreRepository']

# Relations defined in this domain  
RELATIONS_DEFINED: List[str] = ['RelationType', 'RelationStrength', 'Relation', 'RelationGraph', 'RelationMapping']

# Default behavior policies
BEHAVIOR_POLICIES: Dict[str, Any] = {
    "strict_validation": True,
    "auto_dependency_resolution": True,
    "lazy_loading": False,
    "cache_enabled": True,
    "error_handling": "strict",
    "logging_level": "INFO"
}

# Export interface for external access
EXPORT_INTERFACE: List[str] = [
    "get_domain_metadata",
    "validate_configuration", 
    "cost_metadata",
    "DATA_TYPES_AVAILABLE",
    "RELATIONS_DEFINED",
    "BEHAVIOR_POLICIES"
]

def get_domain_metadata() -> DomainConfiguration:
    """
    Get complete domain configuration metadata
    
    Returns:
        DomainConfiguration with all domain metadata and policies
    """
    return DomainConfiguration(
        domain_name="registry",
        cost_metadata=cost_metadata,
        data_types_available=DATA_TYPES_AVAILABLE,
        relations_defined=RELATIONS_DEFINED,
        behavior_policies=BEHAVIOR_POLICIES,
        export_interface=EXPORT_INTERFACE
    )

def validate_configuration() -> bool:
    """
    Validate domain configuration for consistency and completeness
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Validate cost metadata completeness
        required_cost_fields = [
            "priority_index", "compute_time_weight", "exposure_type",
            "dependency_level", "thread_safe", "load_order"
        ]
        
        for field in required_cost_fields:
            if field not in cost_metadata:
                logger.error(f"Missing required cost metadata field: {field}")
                return False
        
        # Validate exposure type
        valid_exposure_types = ["public", "internal", "private"]
        if cost_metadata["exposure_type"] not in valid_exposure_types:
            logger.error(f"Invalid exposure type: {cost_metadata['exposure_type']}")
            return False
        
        # Validate load order consistency
        if not isinstance(cost_metadata["load_order"], int) or cost_metadata["load_order"] < 0:
            logger.error(f"Invalid load order: {cost_metadata['load_order']}")
            return False
        
        logger.info(f"Domain {domain_name} configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def get_cost_metadata() -> DomainCostMetadata:
    """Get domain cost metadata for orchestration"""
    return cost_metadata

def get_behavior_policy(policy_name: str) -> Any:
    """Get specific behavior policy value"""
    return BEHAVIOR_POLICIES.get(policy_name)

def update_behavior_policy(policy_name: str, value: Any) -> bool:
    """Update behavior policy (runtime configuration)"""
    if policy_name in BEHAVIOR_POLICIES:
        BEHAVIOR_POLICIES[policy_name] = value
        logger.info(f"Updated behavior policy {policy_name} = {value}")
        return True
    else:
        logger.warning(f"Unknown behavior policy: {policy_name}")
        return False

# Export all configuration interfaces
__all__ = [
    "cost_metadata",
    "get_domain_metadata",
    "validate_configuration",
    "get_cost_metadata", 
    "get_behavior_policy",
    "update_behavior_policy",
    "DATA_TYPES_AVAILABLE",
    "RELATIONS_DEFINED",
    "BEHAVIOR_POLICIES",
    "DomainCostMetadata",
    "DomainConfiguration"
]

# Auto-validate configuration on module load
if not validate_configuration():
    logger.warning(f"Domain registry configuration loaded with validation warnings")
else:
    logger.debug(f"Domain registry configuration loaded successfully")

# [EOF] - End of registry domain configuration module
