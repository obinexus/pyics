#!/usr/bin/env python3
"""
pyics/core/structures/config.py
Structures Domain Configuration

Generated: 2025-05-31T19:56:14.953871
Engineering Lead: Nnamdi Okpala / OBINexus Computing
Domain: structures

PROBLEM SOLVED: Defines immutable data containers for all calendar operations
DEPENDENCIES: Core domain components only
THREAD SAFETY: Yes - Immutable configuration data
DETERMINISTIC: Yes - Static configuration with predictable behavior

Configuration module providing cost metadata, behavior policies, and domain-specific
settings for the structures domain following DOP compliance principles.
"""

from typing import Dict, List, Any, TypedDict, Literal
import logging

logger = logging.getLogger(f"pyics.core.structures.config")

# Type definitions for domain configuration
class DomainCostMetadata(TypedDict):
    priority_index: int
    compute_time_weight: float
    exposure_type: str
    dependency_level: int
    thread_safe: bool
    load_order: int

class DomainConfiguration(TypedDict):
    domain_name: str
    cost_metadata: DomainCostMetadata
    problem_solved: str
    separation_rationale: str
    merge_potential: str
    behavior_policies: Dict[str, Any]
    export_interface: List[str]

# Cost metadata for structures domain
cost_metadata: DomainCostMetadata = {
    "priority_index": 2,
    "compute_time_weight": 0.2,
    "exposure_type": "version_required",
    "dependency_level": 1,
    "thread_safe": True,
    "load_order": 30
}

# Domain behavior policies
BEHAVIOR_POLICIES: Dict[str, Any] = {
    "strict_validation": True,
    "atomic_operations": false,
    "immutable_structures": true,
    "interface_only": false,
    "error_handling": "strict",
    "logging_level": "INFO",
    "performance_monitoring": True
}

# Export interface definition
EXPORT_INTERFACE: List[str] = [
    "get_domain_metadata",
    "validate_configuration",
    "cost_metadata",
    "BEHAVIOR_POLICIES"
]

def get_domain_metadata() -> DomainConfiguration:
    """
    Get complete domain configuration metadata
    
    Returns:
        DomainConfiguration with all domain metadata and policies
    """
    return DomainConfiguration(
        domain_name="structures",
        cost_metadata=cost_metadata,
        problem_solved="Defines immutable data containers for all calendar operations",
        separation_rationale="Must remain pure (dataclasses only) to enforce data immutability and safety",
        merge_potential="PRESERVE",
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
        required_fields = ["priority_index", "compute_time_weight", "exposure_type", 
                          "dependency_level", "thread_safe", "load_order"]
        
        for field in required_fields:
            if field not in cost_metadata:
                logger.error(f"Missing required cost metadata field: {field}")
                return False
        
        # Validate domain-specific constraints
        if cost_metadata["priority_index"] < 1:
            logger.error("Priority index must be >= 1")
            return False
            
        if cost_metadata["compute_time_weight"] < 0:
            logger.error("Compute time weight cannot be negative")
            return False
        
        logger.info(f"Domain structures configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

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
    "get_behavior_policy",
    "update_behavior_policy",
    "BEHAVIOR_POLICIES",
    "EXPORT_INTERFACE",
    "DomainCostMetadata",
    "DomainConfiguration"
]

# Auto-validate configuration on module load
if not validate_configuration():
    logger.warning(f"Domain structures configuration loaded with validation warnings")
else:
    logger.debug(f"Domain structures configuration loaded successfully")

# [EOF] - End of structures domain configuration module
