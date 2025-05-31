#!/usr/bin/env python3
"""
pyics_architecture_validator.py
Pyics Architecture Validator - Independent Validation Tool

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Standalone validation of Pyics single-pass architecture compliance
Architecture: Independent validation with comprehensive reporting
Methodology: Systematic validation with detailed compliance analysis

PROBLEM SOLVED: Provides comprehensive validation of corrected architecture
DEPENDENCIES: Standard library only (pathlib, ast, importlib)
THREAD SAFETY: Yes - read-only validation operations
DETERMINISTIC: Yes - consistent validation results with detailed reporting

This validator provides independent verification of Pyics architecture compliance,
validating single-pass loading, domain structure, and cost metadata consistency.
"""

import os
import sys
import ast
import json
import time
import importlib
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
import logging

# Configuration
PROJECT_ROOT = Path.cwd()
PYICS_CORE_DIR = "pyics/core"

# Expected domain load order ranges
EXPECTED_LOAD_ORDER = {
    "primitives": (10, 19),
    "protocols": (20, 29), 
    "structures": (30, 39),
    "composition": (40, 49),
    "validators": (50, 59),
    "transformations": (60, 69),
    "registry": (70, 79),
    "routing": (80, 89),
    "safety": (90, 99)
}

# Required modules per domain
REQUIRED_MODULES = [
    "data_types.py",
    "operations.py",
    "relations.py", 
    "config.py",
    "__init__.py"
]

# Prohibited structures (should be cleaned up)
PROHIBITED_STRUCTURES = [
    "implementations",
    "interfaces", 
    "compliance",
    "contracts",
    "tests"
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PyicsArchitectureValidator:
    """
    Comprehensive validator for Pyics single-pass architecture compliance
    
    Provides systematic validation of domain structure, load order compliance,
    cost metadata consistency, and single-pass loading functionality.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.core_dir = self.project_root / PYICS_CORE_DIR
        
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "core_directory_exists": False,
            "domains_discovered": 0,
            "domains_validated": 0,
            "structure_compliance": {},
            "load_order_compliance": {},
            "cost_metadata_compliance": {},
            "single_pass_loading": {},
            "ioc_registry_validation": {},
            "performance_metrics": {},
            "violations": [],
            "recommendations": [],
            "overall_compliance": False
        }
        
        self.discovered_domains = []
        self.domain_metadata = {}
    
    def execute_comprehensive_validation(self) -> Dict[str, Any]:
        """Execute complete architecture validation"""
        logger.info("=" * 60)
        logger.info("PYICS ARCHITECTURE VALIDATION")
        logger.info("Engineering Lead: Nnamdi Okpala / OBINexus Computing") 
        logger.info("=" * 60)
        
        try:
            # Phase 1: Core directory validation
            self._validate_core_directory()
            
            # Phase 2: Domain discovery and analysis
            self._discover_and_analyze_domains()
            
            # Phase 3: Structure compliance validation
            self._validate_structure_compliance()
            
            # Phase 4: Load order compliance validation
            self._validate_load_order_compliance()
            
            # Phase 5: Cost metadata validation
            self._validate_cost_metadata_compliance()
            
            # Phase 6: Single-pass loading validation
            self._validate_single_pass_loading()
            
            # Phase 7: IoC registry validation
            self._validate_ioc_registry()
            
            # Phase 8: Performance metrics collection
            self._collect_performance_metrics()
            
            # Phase 9: Generate recommendations
            self._generate_recommendations()
            
            # Phase 10: Calculate overall compliance
            self._calculate_overall_compliance()
            
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.validation_results["violations"].append(f"Critical validation failure: {e}")
            return self.validation_results
    
    def _validate_core_directory(self) -> None:
        """Validate core directory structure"""
        logger.info("Validating core directory structure...")
        
        if self.core_dir.exists() and self.core_dir.is_dir():
            self.validation_results["core_directory_exists"] = True
            logger.info(f"Core directory found: {self.core_dir}")
        else:
            self.validation_results["core_directory_exists"] = False
            self.validation_results["violations"].append("Core directory missing")
            logger.error(f"Core directory not found: {self.core_dir}")
    
    def _discover_and_analyze_domains(self) -> None:
        """Discover and analyze all domains"""
        logger.info("Discovering and analyzing domains...")
        
        if not self.validation_results["core_directory_exists"]:
            return
        
        for item in self.core_dir.iterdir():
            if (item.is_dir() and 
                not item.name.startswith('.') and 
                not item.name.startswith('_') and
                item.name != '__pycache__'):
                
                domain_name = item.name
                self.discovered_domains.append(domain_name)
                
                # Analyze domain structure
                analysis = self._analyze_domain(item, domain_name)
                self.domain_metadata[domain_name] = analysis
                
                logger.info(f"Analyzed domain: {domain_name}")
        
        self.validation_results["domains_discovered"] = len(self.discovered_domains)
        logger.info(f"Domain discovery complete: {len(self.discovered_domains)} domains found")
    
    def _analyze_domain(self, domain_path: Path, domain_name: str) -> Dict[str, Any]:
        """Analyze individual domain structure and metadata"""
        analysis = {
            "domain_name": domain_name,
            "path": str(domain_path),
            "required_modules_present": [],
            "required_modules_missing": [],
            "prohibited_structures_found": [],
            "config_loadable": False,
            "config_metadata": {},
            "violations": [],
            "compliance_score": 0.0
        }
        
        # Check required modules
        for module in REQUIRED_MODULES:
            module_path = domain_path / module
            if module_path.exists():
                analysis["required_modules_present"].append(module)
            else:
                analysis["required_modules_missing"].append(module)
                analysis["violations"].append(f"missing_module_{module}")
        
        # Check for prohibited structures
        for prohibited in PROHIBITED_STRUCTURES:
            prohibited_path = domain_path / prohibited
            if prohibited_path.exists():
                analysis["prohibited_structures_found"].append(prohibited)
                analysis["violations"].append(f"prohibited_structure_{prohibited}")
        
        # Try to load and validate configuration
        try:
            config_module = importlib.import_module(f"pyics.core.{domain_name}.config")
            analysis["config_loadable"] = True
            
            # Extract configuration metadata
            if hasattr(config_module, 'cost_metadata'):
                analysis["config_metadata"] = config_module.cost_metadata
            
            # Validate configuration
            if hasattr(config_module, 'validate_configuration'):
                if not config_module.validate_configuration():
                    analysis["violations"].append("config_validation_failed")
            else:
                analysis["violations"].append("missing_validate_configuration")
                
        except ImportError as e:
            analysis["violations"].append(f"config_import_failed: {e}")
        except Exception as e:
            analysis["violations"].append(f"config_error: {e}")
        
        # Calculate compliance score
        total_checks = len(REQUIRED_MODULES) + len(PROHIBITED_STRUCTURES) + 2  # +2 for config checks
        violations = len(analysis["violations"])
        analysis["compliance_score"] = max(0.0, (total_checks - violations) / total_checks)
        
        return analysis
    
    def _validate_structure_compliance(self) -> None:
        """Validate domain structure compliance"""
        logger.info("Validating domain structure compliance...")
        
        structure_results = {
            "compliant_domains": [],
            "non_compliant_domains": [],
            "total_violations": 0,
            "compliance_percentage": 0.0
        }
        
        for domain_name, analysis in self.domain_metadata.items():
            violations = analysis["violations"]
            compliance_score = analysis["compliance_score"]
            
            if compliance_score >= 0.9:  # 90% compliance threshold
                structure_results["compliant_domains"].append({
                    "domain": domain_name,
                    "score": compliance_score,
                    "violations": len(violations)
                })
            else:
                structure_results["non_compliant_domains"].append({
                    "domain": domain_name,
                    "score": compliance_score,
                    "violations": violations
                })
            
            structure_results["total_violations"] += len(violations)
        
        # Calculate overall compliance percentage
        if self.discovered_domains:
            compliant_count = len(structure_results["compliant_domains"])
            structure_results["compliance_percentage"] = (compliant_count / len(self.discovered_domains)) * 100
        
        self.validation_results["structure_compliance"] = structure_results
        logger.info(f"Structure compliance: {structure_results['compliance_percentage']:.1f}%")
    
    def _validate_load_order_compliance(self) -> None:
        """Validate load order compliance and dependencies"""
        logger.info("Validating load order compliance...")
        
        load_order_results = {
            "domain_load_orders": {},
            "order_violations": [],
            "dependency_violations": [],
            "load_order_valid": False
        }
        
        # Extract load orders from domain metadata
        for domain_name, analysis in self.domain_metadata.items():
            if "load_order" in analysis["config_metadata"]:
                load_order = analysis["config_metadata"]["load_order"]
                load_order_results["domain_load_orders"][domain_name] = load_order
                
                # Check if load order is in expected range
                if domain_name in EXPECTED_LOAD_ORDER:
                    expected_min, expected_max = EXPECTED_LOAD_ORDER[domain_name]
                    if not (expected_min <= load_order <= expected_max):
                        violation = f"{domain_name}: load_order {load_order} not in expected range [{expected_min}-{expected_max}]"
                        load_order_results["order_violations"].append(violation)
                        self.validation_results["violations"].append(violation)
        
        # Validate load order sequence
        sorted_domains = sorted(load_order_results["domain_load_orders"].items(), key=lambda x: x[1])
        prev_order = 0
        
        for domain, load_order in sorted_domains:
            if load_order < prev_order:
                violation = f"Load order violation: {domain} ({load_order}) < previous ({prev_order})"
                load_order_results["dependency_violations"].append(violation)
                self.validation_results["violations"].append(violation)
            prev_order = load_order
        
        load_order_results["load_order_valid"] = (
            len(load_order_results["order_violations"]) == 0 and
            len(load_order_results["dependency_violations"]) == 0
        )
        
        self.validation_results["load_order_compliance"] = load_order_results
        logger.info(f"Load order compliance: {'✅' if load_order_results['load_order_valid'] else '❌'}")
    
    def _validate_cost_metadata_compliance(self) -> None:
        """Validate cost metadata consistency and completeness"""
        logger.info("Validating cost metadata compliance...")
        
        cost_results = {
            "domains_with_metadata": [],
            "domains_missing_metadata": [],
            "metadata_violations": [],
            "consistency_score": 0.0
        }
        
        required_cost_fields = [
            "priority_index", "compute_time_weight", "exposure_type",
            "dependency_level", "thread_safe", "load_order"
        ]
        
        for domain_name, analysis in self.domain_metadata.items():
            cost_metadata = analysis["config_metadata"]
            
            if cost_metadata:
                cost_results["domains_with_metadata"].append(domain_name)
                
                # Check required fields
                for field in required_cost_fields:
                    if field not in cost_metadata:
                        violation = f"{domain_name}: missing cost metadata field '{field}'"
                        cost_results["metadata_violations"].append(violation)
                        self.validation_results["violations"].append(violation)
                
                # Validate field types and ranges
                if "priority_index" in cost_metadata:
                    priority = cost_metadata["priority_index"]
                    if not isinstance(priority, int) or priority < 1 or priority > 10:
                        violation = f"{domain_name}: invalid priority_index {priority} (must be 1-10)"
                        cost_results["metadata_violations"].append(violation)
                
                if "compute_time_weight" in cost_metadata:
                    weight = cost_metadata["compute_time_weight"]
                    if not isinstance(weight, (int, float)) or weight < 0 or weight > 10:
                        violation = f"{domain_name}: invalid compute_time_weight {weight} (must be 0-10)"
                        cost_results["metadata_violations"].append(violation)
                
                if "exposure_type" in cost_metadata:
                    exposure = cost_metadata["exposure_type"]
                    valid_exposures = ["public", "internal", "private"]
                    if exposure not in valid_exposures:
                        violation = f"{domain_name}: invalid exposure_type '{exposure}' (must be {valid_exposures})"
                        cost_results["metadata_violations"].append(violation)
                
            else:
                cost_results["domains_missing_metadata"].append(domain_name)
                violation = f"{domain_name}: missing cost metadata"
                cost_results["metadata_violations"].append(violation)
                self.validation_results["violations"].append(violation)
        
        # Calculate consistency score
        total_domains = len(self.discovered_domains)
        domains_with_valid_metadata = total_domains - len(cost_results["metadata_violations"])
        cost_results["consistency_score"] = (domains_with_valid_metadata / total_domains * 100) if total_domains > 0 else 0
        
        self.validation_results["cost_metadata_compliance"] = cost_results
        logger.info(f"Cost metadata compliance: {cost_results['consistency_score']:.1f}%")
    
    def _validate_single_pass_loading(self) -> None:
        """Validate single-pass loading functionality"""
        logger.info("Validating single-pass loading...")
        
        loading_results = {
            "loading_successful": False,
            "load_time": 0.0,
            "domains_loaded": [],
            "loading_errors": [],
            "performance_acceptable": False
        }
        
        try:
            start_time = time.time()
            
            # Test IoC registry import and initialization
            ioc_module = importlib.import_module("pyics.core.ioc_registry")
            
            if hasattr(ioc_module, 'get_registry'):
                registry = ioc_module.get_registry()
                
                # Test domain loading
                if hasattr(ioc_module, 'get_all_domains'):
                    domains = ioc_module.get_all_domains()
                    loading_results["domains_loaded"] = domains
                    
                    # Validate each domain loaded correctly
                    for domain in domains:
                        try:
                            domain_module = importlib.import_module(f"pyics.core.{domain}")
                            if hasattr(domain_module, 'get_domain_metadata'):
                                metadata = domain_module.get_domain_metadata()
                                if not metadata:
                                    loading_results["loading_errors"].append(f"{domain}: metadata retrieval failed")
                        except ImportError as e:
                            loading_results["loading_errors"].append(f"{domain}: import failed - {e}")
                
                # Test architecture validation
                if hasattr(ioc_module, 'validate_architecture'):
                    if not ioc_module.validate_architecture():
                        loading_results["loading_errors"].append("Architecture validation failed")
                
                load_time = time.time() - start_time
                loading_results["load_time"] = load_time
                loading_results["performance_acceptable"] = load_time < 1.0  # Less than 1 second
                
                loading_results["loading_successful"] = len(loading_results["loading_errors"]) == 0
                
            else:
                loading_results["loading_errors"].append("IoC registry missing get_registry method")
                
        except ImportError as e:
            loading_results["loading_errors"].append(f"IoC registry import failed: {e}")
        except Exception as e:
            loading_results["loading_errors"].append(f"Single-pass loading failed: {e}")
        
        self.validation_results["single_pass_loading"] = loading_results
        logger.info(f"Single-pass loading: {'✅' if loading_results['loading_successful'] else '❌'}")
    
    def _validate_ioc_registry(self) -> None:
        """Validate IoC registry implementation"""
        logger.info("Validating IoC registry...")
        
        registry_results = {
            "registry_exists": False,
            "registry_functional": False,
            "required_methods_present": [],
            "required_methods_missing": [],
            "validation_errors": []
        }
        
        registry_path = self.core_dir / "ioc_registry.py"
        
        if registry_path.exists():
            registry_results["registry_exists"] = True
            
            try:
                ioc_module = importlib.import_module("pyics.core.ioc_registry")
                
                # Check required methods
                required_methods = [
                    "get_registry", "get_domain_metadata", "get_all_domains",
                    "validate_architecture", "SINGLE_PASS_LOAD_ORDER"
                ]
                
                for method in required_methods:
                    if hasattr(ioc_module, method):
                        registry_results["required_methods_present"].append(method)
                    else:
                        registry_results["required_methods_missing"].append(method)
                        registry_results["validation_errors"].append(f"Missing required method: {method}")
                
                # Test registry functionality
                if hasattr(ioc_module, 'get_registry'):
                    try:
                        registry = ioc_module.get_registry()
                        if registry:
                            registry_results["registry_functional"] = True
                        else:
                            registry_results["validation_errors"].append("Registry initialization returned None")
                    except Exception as e:
                        registry_results["validation_errors"].append(f"Registry initialization failed: {e}")
                
            except ImportError as e:
                registry_results["validation_errors"].append(f"Registry import failed: {e}")
        else:
            registry_results["validation_errors"].append("IoC registry file not found")
        
        self.validation_results["ioc_registry_validation"] = registry_results
        logger.info(f"IoC registry validation: {'✅' if registry_results['registry_functional'] else '❌'}")
    
    def _collect_performance_metrics(self) -> None:
        """Collect performance metrics for optimization analysis"""
        logger.info("Collecting performance metrics...")
        
        performance_results = {
            "total_domains": len(self.discovered_domains),
            "load_time_per_domain": {},
            "total_load_time": 0.0,
            "memory_efficient": True,
            "optimization_opportunities": []
        }
        
        try:
            # Get load performance from registry
            ioc_module = importlib.import_module("pyics.core.ioc_registry")
            if hasattr(ioc_module, 'get_registry'):
                registry = ioc_module.get_registry()
                if hasattr(registry, 'get_load_performance'):
                    load_performance = registry.get_load_performance()
                    performance_results["load_time_per_domain"] = load_performance
                    performance_results["total_load_time"] = sum(load_performance.values())
                    
                    # Identify slow domains
                    for domain, load_time in load_performance.items():
                        if load_time > 0.1:  # More than 100ms
                            performance_results["optimization_opportunities"].append(
                                f"{domain}: slow loading ({load_time:.3f}s)"
                            )
                    
                    # Check if total load time is acceptable
                    if performance_results["total_load_time"] > 2.0:  # More than 2 seconds
                        performance_results["optimization_opportunities"].append(
                            f"Total load time high: {performance_results['total_load_time']:.3f}s"
                        )
        
        except Exception as e:
            performance_results["optimization_opportunities"].append(f"Performance metrics unavailable: {e}")
        
        self.validation_results["performance_metrics"] = performance_results
        logger.info(f"Performance analysis complete: {performance_results['total_load_time']:.3f}s total")
    
    def _generate_recommendations(self) -> None:
        """Generate recommendations based on validation results"""
        logger.info("Generating recommendations...")
        
        recommendations = []
        
        # Structure compliance recommendations
        structure_compliance = self.validation_results["structure_compliance"]
        if structure_compliance["compliance_percentage"] < 100:
            non_compliant = structure_compliance["non_compliant_domains"]
            recommendations.append(f"Fix structure violations in {len(non_compliant)} domains")
            
            for domain_info in non_compliant:
                domain = domain_info["domain"]
                recommendations.append(f"  • {domain}: address {len(domain_info['violations'])} violations")
        
        # Load order recommendations
        load_order = self.validation_results["load_order_compliance"]
        if not load_order["load_order_valid"]:
            recommendations.append("Fix load order violations")
            for violation in load_order["order_violations"]:
                recommendations.append(f"  • {violation}")
        
        # Cost metadata recommendations
        cost_compliance = self.validation_results["cost_metadata_compliance"]
        if cost_compliance["consistency_score"] < 100:
            missing_domains = cost_compliance["domains_missing_metadata"]
            if missing_domains:
                recommendations.append(f"Add cost metadata to {len(missing_domains)} domains")
        
        # Single-pass loading recommendations
        loading = self.validation_results["single_pass_loading"]
        if not loading["loading_successful"]:
            recommendations.append("Fix single-pass loading issues")
            for error in loading["loading_errors"]:
                recommendations.append(f"  • {error}")
        
        # Performance recommendations
        performance = self.validation_results["performance_metrics"]
        if performance["optimization_opportunities"]:
            recommendations.append("Performance optimization opportunities:")
            for opportunity in performance["optimization_opportunities"]:
                recommendations.append(f"  • {opportunity}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Architecture is fully compliant - no actions required")
        else:
            recommendations.insert(0, "Critical issues require attention before production use")
        
        self.validation_results["recommendations"] = recommendations
        logger.info(f"Generated {len(recommendations)} recommendations")
    
    def _calculate_overall_compliance(self) -> None:
        """Calculate overall architecture compliance score"""
        logger.info("Calculating overall compliance...")
        
        compliance_factors = {
            "structure": self.validation_results["structure_compliance"]["compliance_percentage"],
            "load_order": 100 if self.validation_results["load_order_compliance"]["load_order_valid"] else 0,
            "cost_metadata": self.validation_results["cost_metadata_compliance"]["consistency_score"],
            "single_pass_loading": 100 if self.validation_results["single_pass_loading"]["loading_successful"] else 0,
            "ioc_registry": 100 if self.validation_results["ioc_registry_validation"]["registry_functional"] else 0
        }
        
        # Weighted average (all factors equally important)
        overall_score = sum(compliance_factors.values()) / len(compliance_factors)
        
        # Overall compliance requires 95% score
        self.validation_results["overall_compliance"] = overall_score >= 95.0
        self.validation_results["compliance_score"] = overall_score
        
        logger.info(f"Overall compliance: {overall_score:.1f}% ({'✅ PASS' if overall_score >= 95 else '❌ FAIL'})")
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        results = self.validation_results
        
        report = f"""
# Pyics Architecture Validation Report

**Generated**: {results['timestamp']}
**Project Root**: {results['project_root']}
**Engineering Lead**: Nnamdi Okpala / OBINexus Computing

## Executive Summary

- **Overall Compliance**: {results.get('compliance_score', 0):.1f}% ({'✅ PASS' if results.get('overall_compliance', False) else '❌ FAIL'})
- **Domains Discovered**: {results['domains_discovered']}
- **Core Directory**: {'✅ Found' if results['core_directory_exists'] else '❌ Missing'}
- **Total Violations**: {len(results['violations'])}

## Detailed Results

### 1. Structure Compliance
- **Compliance Rate**: {results['structure_compliance'].get('compliance_percentage', 0):.1f}%
- **Compliant Domains**: {len(results['structure_compliance'].get('compliant_domains', []))}
- **Non-Compliant Domains**: {len(results['structure_compliance'].get('non_compliant_domains', []))}

### 2. Load Order Compliance
- **Load Order Valid**: {'✅' if results['load_order_compliance'].get('load_order_valid', False) else '❌'}
- **Order Violations**: {len(results['load_order_compliance'].get('order_violations', []))}
- **Dependency Violations**: {len(results['load_order_compliance'].get('dependency_violations', []))}

### 3. Cost Metadata Compliance
- **Consistency Score**: {results['cost_metadata_compliance'].get('consistency_score', 0):.1f}%
- **Domains with Metadata**: {len(results['cost_metadata_compliance'].get('domains_with_metadata', []))}
- **Missing Metadata**: {len(results['cost_metadata_compliance'].get('domains_missing_metadata', []))}

### 4. Single-Pass Loading
- **Loading Successful**: {'✅' if results['single_pass_loading'].get('loading_successful', False) else '❌'}
- **Load Time**: {results['single_pass_loading'].get('load_time', 0):.3f}s
- **Domains Loaded**: {len(results['single_pass_loading'].get('domains_loaded', []))}
- **Loading Errors**: {len(results['single_pass_loading'].get('loading_errors', []))}

### 5. IoC Registry Validation
- **Registry Exists**: {'✅' if results['ioc_registry_validation'].get('registry_exists', False) else '❌'}
- **Registry Functional**: {'✅' if results['ioc_registry_validation'].get('registry_functional', False) else '❌'}
- **Required Methods Present**: {len(results['ioc_registry_validation'].get('required_methods_present', []))}
- **Required Methods Missing**: {len(results['ioc_registry_validation'].get('required_methods_missing', []))}

### 6. Performance Metrics
- **Total Load Time**: {results['performance_metrics'].get('total_load_time', 0):.3f}s
- **Optimization Opportunities**: {len(results['performance_metrics'].get('optimization_opportunities', []))}

## Violations Found
"""
        
        if results['violations']:
            for i, violation in enumerate(results['violations'], 1):
                report += f"\n{i}. {violation}"
        else:
            report += "\nNo violations found - architecture is compliant."
        
        report += "\n\n## Recommendations\n"
        
        for i, recommendation in enumerate(results.get('recommendations', []), 1):
            report += f"\n{i}. {recommendation}"
        
        report += f"""

## Next Steps

{'✅ Architecture is production-ready' if results.get('overall_compliance', False) else '❌ Fix violations before production use'}

### Immediate Actions Required:
"""
        
        if not results.get('overall_compliance', False):
            priority_actions = [
                rec for rec in results.get('recommendations', [])
                if any(keyword in rec.lower() for keyword in ['critical', 'fix', 'missing', 'failed'])
            ]
            
            for action in priority_actions[:5]:  # Top 5 priority actions
                report += f"\n- {action}"
        else:
            report += "\n- No immediate actions required"
            report += "\n- Consider performance optimizations if needed"
            report += "\n- Maintain architecture compliance in future changes"
        
        report += f"""

---
**Report Generated**: {results['timestamp']}
**Validation Tool**: Pyics Architecture Validator v1.0.0
**Engineering Lead**: Nnamdi Okpala / OBINexus Computing
"""
        
        return report

def main():
    """Main execution function"""
    validator = PyicsArchitectureValidator(PROJECT_ROOT)
    results = validator.execute_comprehensive_validation()
    
    # Generate and display report
    report = validator.generate_validation_report()
    
    # Save report to file
    report_path = PROJECT_ROOT / f"pyics_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Display summary
    print("=" * 80)
    print("🎯 PYICS ARCHITECTURE VALIDATION COMPLETE")
    print("=" * 80)
    print(f"📊 Overall Compliance: {results.get('compliance_score', 0):.1f}%")
    print(f"📁 Domains Analyzed: {results['domains_discovered']}")
    print(f"⚠️  Total Violations: {len(results['violations'])}")
    print(f"📋 Recommendations: {len(results.get('recommendations', []))}")
    print(f"📄 Report Saved: {report_path}")
    print("=" * 80)
    
    if results.get('overall_compliance', False):
        print("🎉 VALIDATION PASSED - Architecture is compliant!")
        print("✅ Single-pass loading working correctly")
        print("✅ Domain structure follows standards")
        print("✅ Cost metadata properly configured")
        print("✅ IoC registry functional")
    else:
        print("❌ VALIDATION FAILED - Architecture needs fixes")
        print("\n🔧 Priority fixes needed:")
        for i, rec in enumerate(results.get('recommendations', [])[:3], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n📄 See full report: {report_path}")
    
    print("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if results.get('overall_compliance', False) else 1)

if __name__ == "__main__":
    main()

# [EOF] - End of Pyics architecture validator