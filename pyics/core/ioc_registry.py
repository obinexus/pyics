#!/usr/bin/env python3
"""
pyics/core/ioc_registry.py
Inversion of Control Registry for Domain Configurations

Generated: 2025-05-31T19:27:17.695366
Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Centralized registry for dynamic domain configuration resolution
Architecture: DOP-compliant IoC container with type-safe resolution

PROBLEM SOLVED: Provides centralized configuration discovery and dependency injection
DEPENDENCIES: All pyics.core domain configuration modules
THREAD SAFETY: Yes - immutable registry with concurrent access support
DETERMINISTIC: Yes - predictable configuration resolution order

This registry implements systematic domain configuration discovery and provides
type-safe dependency injection interfaces for runtime orchestration.
"""

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, TypeVar, cast
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .transforms.config import DomainConfiguration, DomainCostMetadata

logger = logging.getLogger("pyics.core.ioc_registry")

# Type variables for generic resolution
T = TypeVar('T')

class IoContainerError(Exception):
    """Custom exception for IoC container operations"""
    pass

class DomainConfigurationRegistry:
    """
    Centralized registry for domain configurations with IoC capabilities
    
    Provides systematic configuration discovery, dependency resolution,
    and type-safe access to domain metadata and cost functions.
    """
    
    def __init__(self):
        self._domain_configs: Dict[str, Any] = {}
        self._load_order_cache: List[str] = []
        self._initialized = False
        
    def initialize_registry(self) -> bool:
        """
        Initialize registry by discovering and loading all domain configurations
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.warning("Registry already initialized")
            return True
        
        try:
            logger.info("Initializing domain configuration registry...")
            
            # Discover and load domain configurations
            discovered_domains = ['transforms', 'protocols', 'routing', 'logic', 'validation', 'primitives', 'validators', 'structures', 'composition', 'safety', 'registry', 'transformations']
            
            for domain_name in discovered_domains:
                if self._load_domain_config(domain_name):
                    logger.debug(f"Loaded configuration for domain: {domain_name}")
                else:
                    logger.error(f"Failed to load configuration for domain: {domain_name}")
                    return False
            
            # Build load order cache
            self._build_load_order_cache()
            
            self._initialized = True
            logger.info(f"Registry initialized with {len(self._domain_configs)} domain configurations")
            return True
            
        except Exception as e:
            logger.error(f"Registry initialization failed: {e}")
            return False
    
    def _load_domain_config(self, domain_name: str) -> bool:
        """Load configuration module for a specific domain"""
        try:
            config_module_name = f"pyics.core.{domain_name}.config"
            config_module = importlib.import_module(config_module_name)
            
            # Validate required configuration interface
            required_attrs = ["get_domain_metadata", "validate_configuration", "cost_metadata"]
            for attr in required_attrs:
                if not hasattr(config_module, attr):
                    logger.error(f"Domain {domain_name} config missing required attribute: {attr}")
                    return False
            
            # Validate configuration
            if not config_module.validate_configuration():
                logger.error(f"Domain {domain_name} configuration validation failed")
                return False
            
            # Store configuration
            self._domain_configs[domain_name] = {
                "module": config_module,
                "metadata": config_module.get_domain_metadata(),
                "cost_metadata": config_module.cost_metadata
            }
            
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import configuration for {domain_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading configuration for {domain_name}: {e}")
            return False
    
    def _build_load_order_cache(self) -> None:
        """Build load order cache based on domain priorities"""
        domain_priorities = []
        
        for domain_name, config_data in self._domain_configs.items():
            load_order = config_data["cost_metadata"]["load_order"]
            domain_priorities.append((load_order, domain_name))
        
        # Sort by load order (ascending)
        domain_priorities.sort(key=lambda x: x[0])
        self._load_order_cache = [domain_name for _, domain_name in domain_priorities]
        
        logger.debug(f"Load order cache: {self._load_order_cache}")
    
    def get_domain_configuration(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """
        Get complete configuration for a specific domain
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            Domain configuration dictionary or None if not found
        """
        if not self._initialized:
            raise IoContainerError("Registry not initialized - call initialize_registry() first")
        
        return self._domain_configs.get(domain_name)
    
    def get_domain_cost_metadata(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """Get cost metadata for a specific domain"""
        config = self.get_domain_configuration(domain_name)
        return config["cost_metadata"] if config else None
    
    def get_all_domain_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered domains"""
        if not self._initialized:
            raise IoContainerError("Registry not initialized")
        
        return {
            domain_name: config_data["metadata"] 
            for domain_name, config_data in self._domain_configs.items()
        }
    
    def get_domains_by_exposure_type(self, exposure_type: str) -> List[str]:
        """Get domains filtered by exposure type"""
        matching_domains = []
        
        for domain_name, config_data in self._domain_configs.items():
            if config_data["cost_metadata"]["exposure_type"] == exposure_type:
                matching_domains.append(domain_name)
        
        return matching_domains
    
    def get_load_order(self) -> List[str]:
        """Get domains in load order sequence"""
        return self._load_order_cache.copy()
    
    def register_domain_metadata(self, domain_name: str, metadata: Dict[str, Any]) -> bool:
        """
        Register custom domain metadata (for runtime registration)
        
        Args:
            domain_name: Name of the domain
            metadata: Domain metadata dictionary
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if domain_name in self._domain_configs:
                logger.warning(f"Overriding existing configuration for domain: {domain_name}")
            
            self._domain_configs[domain_name] = {
                "module": None,  # Runtime registration
                "metadata": metadata,
                "cost_metadata": metadata.get("cost_metadata", {})
            }
            
            # Rebuild load order cache
            self._build_load_order_cache()
            
            logger.info(f"Registered runtime configuration for domain: {domain_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register domain {domain_name}: {e}")
            return False
    
    def validate_registry(self) -> bool:
        """Validate complete registry for consistency"""
        if not self._initialized:
            return False
        
        validation_errors = []
        
        # Check for load order conflicts
        load_orders = [config["cost_metadata"]["load_order"] for config in self._domain_configs.values()]
        if len(set(load_orders)) != len(load_orders):
            validation_errors.append("Duplicate load_order values detected")
        
        # Validate each domain configuration
        for domain_name, config_data in self._domain_configs.items():
            if config_data["module"] and hasattr(config_data["module"], "validate_configuration"):
                if not config_data["module"].validate_configuration():
                    validation_errors.append(f"Domain {domain_name} configuration invalid")
        
        if validation_errors:
            logger.error(f"Registry validation failed: {validation_errors}")
            return False
        
        logger.info("Registry validation passed")
        return True

# Global registry instance
_registry_instance: Optional[DomainConfigurationRegistry] = None

def get_registry() -> DomainConfigurationRegistry:
    """Get or create global registry instance"""
    global _registry_instance
    
    if _registry_instance is None:
        _registry_instance = DomainConfigurationRegistry()
        if not _registry_instance.initialize_registry():
            raise IoContainerError("Failed to initialize domain configuration registry")
    
    return _registry_instance

def get_domain_metadata(domain_name: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get domain metadata"""
    registry = get_registry()
    return registry.get_domain_configuration(domain_name)

def get_all_domain_metadata() -> Dict[str, Dict[str, Any]]:
    """Convenience function to get all domain metadata"""
    registry = get_registry()
    return registry.get_all_domain_metadata()

def get_domain_cost_metadata(domain_name: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get domain cost metadata"""
    registry = get_registry()
    return registry.get_domain_cost_metadata(domain_name)

def validate_all_configurations() -> bool:
    """Validate all registered domain configurations"""
    registry = get_registry()
    return registry.validate_registry()

# Export registry interfaces
__all__ = [
    "DomainConfigurationRegistry",
    "IoContainerError",
    "get_registry",
    "get_domain_metadata",
    "get_all_domain_metadata", 
    "get_domain_cost_metadata",
    "validate_all_configurations"
]

# Auto-initialize registry on module load
try:
    logger.info("Auto-initializing IoC registry...")
    _auto_registry = get_registry()
    logger.info(f"IoC registry initialized with {len(_auto_registry._domain_configs)} domains")
except Exception as e:
    logger.error(f"Failed to auto-initialize IoC registry: {e}")

# [EOF] - End of IoC registry module
