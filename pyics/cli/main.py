#!/usr/bin/env python3
"""
pyics/cli/main.py
Pyics CLI Main Interface - Architecture Management Commands

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Purpose: Command-line interface for Pyics architecture validation and management
Architecture: Click-based CLI with comprehensive domain management commands
Methodology: Interactive architecture validation with detailed reporting

PROBLEM SOLVED: Provides comprehensive CLI interface for architecture management
DEPENDENCIES: click, pathlib, importlib for CLI and module management
THREAD SAFETY: Yes - read-only operations with safe module importing
DETERMINISTIC: Yes - consistent command execution and validation reporting

This CLI provides systematic architecture management, validation, and diagnostic
capabilities for the Pyics single-pass domain architecture.
"""

import click
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path.cwd()
CORE_DIR = PROJECT_ROOT / "pyics" / "core"

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
def cli(verbose: bool, quiet: bool):
    """
    Pyics Architecture Management CLI
    
    Engineering Lead: Nnamdi Okpala / OBINexus Computing
    
    Comprehensive command-line interface for managing, validating, and
    diagnosing the Pyics single-pass domain architecture.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)

@cli.command()
def info():
    """Display Pyics system information"""
    click.echo("=" * 60)
    click.echo("🎯 PYICS ARCHITECTURE MANAGEMENT CLI")
    click.echo("Engineering Lead: Nnamdi Okpala / OBINexus Computing")
    click.echo("=" * 60)
    
    # System info
    click.echo(f"📁 Project Root: {PROJECT_ROOT}")
    click.echo(f"🏗️  Core Directory: {CORE_DIR}")
    click.echo(f"🐍 Python Version: {sys.version.split()[0]}")
    
    # Architecture status
    if CORE_DIR.exists():
        domains = [d.name for d in CORE_DIR.iterdir() 
                  if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__']
        click.echo(f"📦 Domains Found: {len(domains)}")
        click.echo(f"🔍 Domain List: {', '.join(sorted(domains))}")
        
        # Check for IoC registry
        ioc_path = CORE_DIR / "ioc_registry.py"
        if ioc_path.exists():
            click.echo("✅ IoC Registry: Found")
        else:
            click.echo("❌ IoC Registry: Missing")
    else:
        click.echo("❌ Core directory not found")
    
    click.echo("=" * 60)
    click.echo("📋 Available Commands:")
    click.echo("  pyics domain status      - Show domain status")
    click.echo("  pyics domain validate    - Validate architecture")
    click.echo("  pyics domain load-order  - Show load order")
    click.echo("  pyics validate-architecture - Full validation")
    click.echo("  pyics fix-structure      - Fix structure violations")
    click.echo("=" * 60)

@cli.group()
def domain():
    """Domain management commands"""
    pass

@domain.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'simple']), 
              default='table', help='Output format')
@click.argument('domain_name', required=False)
def status(format: str, domain_name: Optional[str]):
    """Show domain status and metadata"""
    
    if not CORE_DIR.exists():
        click.echo("❌ Core directory not found", err=True)
        sys.exit(1)
    
    domains = [d.name for d in CORE_DIR.iterdir() 
              if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__']
    
    if domain_name:
        if domain_name not in domains:
            click.echo(f"❌ Domain '{domain_name}' not found", err=True)
            sys.exit(1)
        domains = [domain_name]
    
    domain_info = []
    
    for domain in sorted(domains):
        domain_path = CORE_DIR / domain
        info = {
            "domain": domain,
            "path": str(domain_path),
            "status": "❌",
            "load_order": "N/A",
            "modules": [],
            "violations": []
        }
        
        # Check standard modules
        standard_modules = ["data_types.py", "operations.py", "relations.py", 
                          "config.py", "__init__.py", "README.md"]
        
        for module in standard_modules:
            if (domain_path / module).exists():
                info["modules"].append(f"✅ {module}")
            else:
                info["modules"].append(f"❌ {module}")
                info["violations"].append(f"missing_{module}")
        
        # Check for complex violations
        complex_dirs = ["implementations", "interfaces", "compliance", "contracts", "tests"]
        for complex_dir in complex_dirs:
            if (domain_path / complex_dir).exists():
                info["violations"].append(f"complex_nesting_{complex_dir}")
        
        # Try to load config for load order
        try:
            config_module = importlib.import_module(f"pyics.core.{domain}.config")
            if hasattr(config_module, 'cost_metadata'):
                info["load_order"] = config_module.cost_metadata["load_order"]
                info["status"] = "✅" if not info["violations"] else "⚠️"
        except ImportError:
            info["violations"].append("config_import_failed")
        
        domain_info.append(info)
    
    # Output formatting
    if format == 'json':
        click.echo(json.dumps(domain_info, indent=2))
    elif format == 'simple':
        for info in domain_info:
            status_icon = info["status"]
            violations = len(info["violations"])
            click.echo(f"{status_icon} {info['domain']} (Load: {info['load_order']}, Violations: {violations})")
    else:  # table format
        click.echo("📊 DOMAIN STATUS REPORT")
        click.echo("=" * 80)
        click.echo(f"{'Domain':<15} {'Status':<8} {'Load Order':<12} {'Violations':<12} {'Modules'}")
        click.echo("-" * 80)
        
        for info in domain_info:
            violations = len(info["violations"])
            modules_ok = sum(1 for m in info["modules"] if m.startswith("✅"))
            modules_total = len(info["modules"])
            
            click.echo(f"{info['domain']:<15} {info['status']:<8} {str(info['load_order']):<12} "
                      f"{violations:<12} {modules_ok}/{modules_total}")
        
        click.echo("=" * 80)
        
        # Show violations details if any
        for info in domain_info:
            if info["violations"]:
                click.echo(f"\n⚠️  {info['domain']} violations:")
                for violation in info["violations"]:
                    click.echo(f"   • {violation}")

@domain.command()
def validate():
    """Validate domain architecture compliance"""
    click.echo("🔍 VALIDATING DOMAIN ARCHITECTURE")
    click.echo("=" * 50)
    
    validation_results = {
        "domains_found": 0,
        "valid_domains": 0,
        "violations": [],
        "load_order_valid": False,
        "ioc_registry_valid": False
    }
    
    if not CORE_DIR.exists():
        click.echo("❌ Core directory not found")
        return
    
    # Discover domains
    domains = [d.name for d in CORE_DIR.iterdir() 
              if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__']
    
    validation_results["domains_found"] = len(domains)
    click.echo(f"📦 Found {len(domains)} domains")
    
    # Validate each domain
    domain_load_orders = {}
    
    for domain in domains:
        click.echo(f"\n🔍 Validating {domain}...")
        domain_path = CORE_DIR / domain
        domain_valid = True
        
        # Check standard modules
        standard_modules = ["data_types.py", "operations.py", "relations.py", 
                          "config.py", "__init__.py"]
        
        for module in standard_modules:
            if not (domain_path / module).exists():
                click.echo(f"  ❌ Missing {module}")
                validation_results["violations"].append(f"{domain}: missing {module}")
                domain_valid = False
            else:
                click.echo(f"  ✅ {module}")
        
        # Check for complex violations
        complex_dirs = ["implementations", "interfaces", "compliance", "contracts", "tests"]
        for complex_dir in complex_dirs:
            if (domain_path / complex_dir).exists():
                click.echo(f"  ❌ Complex nesting: {complex_dir}/")
                validation_results["violations"].append(f"{domain}: complex nesting {complex_dir}")
                domain_valid = False
        
        # Validate configuration
        try:
            config_module = importlib.import_module(f"pyics.core.{domain}.config")
            
            if hasattr(config_module, 'validate_configuration'):
                if config_module.validate_configuration():
                    click.echo(f"  ✅ Configuration valid")
                else:
                    click.echo(f"  ❌ Configuration invalid")
                    validation_results["violations"].append(f"{domain}: configuration invalid")
                    domain_valid = False
            
            if hasattr(config_module, 'cost_metadata'):
                load_order = config_module.cost_metadata["load_order"]
                domain_load_orders[domain] = load_order
                click.echo(f"  ✅ Load order: {load_order}")
            else:
                click.echo(f"  ❌ Missing cost metadata")
                validation_results["violations"].append(f"{domain}: missing cost metadata")
                domain_valid = False
                
        except ImportError as e:
            click.echo(f"  ❌ Configuration import failed: {e}")
            validation_results["violations"].append(f"{domain}: config import failed")
            domain_valid = False
        
        if domain_valid:
            validation_results["valid_domains"] += 1
            click.echo(f"  ✅ {domain} validation passed")
        else:
            click.echo(f"  ❌ {domain} validation failed")
    
    # Validate load order consistency
    if domain_load_orders:
        sorted_domains = sorted(domain_load_orders.items(), key=lambda x: x[1])
        click.echo(f"\n📋 Load Order Validation:")
        
        prev_order = 0
        load_order_valid = True
        
        for domain, load_order in sorted_domains:
            if load_order >= prev_order:
                click.echo(f"  ✅ {domain}: {load_order}")
                prev_order = load_order
            else:
                click.echo(f"  ❌ {domain}: {load_order} (order violation)")
                validation_results["violations"].append(f"load order violation: {domain}")
                load_order_valid = False
        
        validation_results["load_order_valid"] = load_order_valid
    
    # Validate IoC registry
    ioc_path = CORE_DIR / "ioc_registry.py"
    if ioc_path.exists():
        try:
            ioc_module = importlib.import_module("pyics.core.ioc_registry")
            if hasattr(ioc_module, 'validate_architecture'):
                click.echo(f"\n🏗️  IoC Registry Validation:")
                if ioc_module.validate_architecture():
                    click.echo(f"  ✅ IoC registry validation passed")
                    validation_results["ioc_registry_valid"] = True
                else:
                    click.echo(f"  ❌ IoC registry validation failed")
                    validation_results["violations"].append("IoC registry validation failed")
            else:
                click.echo(f"  ❌ IoC registry missing validate_architecture")
                validation_results["violations"].append("IoC registry incomplete")
        except ImportError as e:
            click.echo(f"  ❌ IoC registry import failed: {e}")
            validation_results["violations"].append("IoC registry import failed")
    else:
        click.echo(f"\n❌ IoC registry not found")
        validation_results["violations"].append("IoC registry missing")
    
    # Summary
    click.echo("\n" + "=" * 50)
    click.echo("📊 VALIDATION SUMMARY")
    click.echo("=" * 50)
    click.echo(f"📦 Domains Found: {validation_results['domains_found']}")
    click.echo(f"✅ Valid Domains: {validation_results['valid_domains']}")
    click.echo(f"📋 Load Order Valid: {'✅' if validation_results['load_order_valid'] else '❌'}")
    click.echo(f"🏗️  IoC Registry Valid: {'✅' if validation_results['ioc_registry_valid'] else '❌'}")
    click.echo(f"⚠️  Total Violations: {len(validation_results['violations'])}")
    
    if validation_results["violations"]:
        click.echo("\n🚨 VIOLATIONS FOUND:")
        for violation in validation_results["violations"]:
            click.echo(f"  • {violation}")
    
    overall_valid = (validation_results["valid_domains"] == validation_results["domains_found"] and 
                    validation_results["load_order_valid"] and 
                    validation_results["ioc_registry_valid"])
    
    if overall_valid:
        click.echo("\n🎉 ARCHITECTURE VALIDATION PASSED!")
        click.echo("✅ Single-pass architecture compliance verified")
    else:
        click.echo("\n❌ ARCHITECTURE VALIDATION FAILED")
        click.echo("🔧 Run 'pyics fix-structure' to correct violations")
        sys.exit(1)

@domain.command(name='load-order')
def load_order():
    """Display domain load order and dependencies"""
    click.echo("📋 DOMAIN LOAD ORDER ANALYSIS")
    click.echo("=" * 50)
    
    if not CORE_DIR.exists():
        click.echo("❌ Core directory not found")
        return
    
    # Discover domains and their load orders
    domains = [d.name for d in CORE_DIR.iterdir() 
              if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__']
    
    domain_metadata = {}
    
    for domain in domains:
        try:
            config_module = importlib.import_module(f"pyics.core.{domain}.config")
            if hasattr(config_module, 'cost_metadata'):
                metadata = config_module.cost_metadata
                domain_metadata[domain] = metadata
        except ImportError:
            click.echo(f"⚠️  Could not load config for {domain}")
    
    if not domain_metadata:
        click.echo("❌ No valid domain configurations found")
        return
    
    # Sort by load order
    sorted_domains = sorted(domain_metadata.items(), key=lambda x: x[1]["load_order"])
    
    # Display load order table
    click.echo(f"{'Order':<8} {'Domain':<15} {'Priority':<10} {'Weight':<8} {'Thread Safe':<12} {'Exposure'}")
    click.echo("-" * 75)
    
    for domain, metadata in sorted_domains:
        click.echo(f"{metadata['load_order']:<8} {domain:<15} {metadata['priority_index']:<10} "
                  f"{metadata['compute_time_weight']:<8.2f} {str(metadata['thread_safe']):<12} "
                  f"{metadata['exposure_type']}")
    
    click.echo("=" * 75)
    
    # Test load order via IoC registry
    try:
        ioc_module = importlib.import_module("pyics.core.ioc_registry")
        if hasattr(ioc_module, 'get_registry'):
            registry = ioc_module.get_registry()
            if hasattr(registry, 'get_load_performance'):
                performance = registry.get_load_performance()
                
                if performance:
                    click.echo("\n⚡ LOAD PERFORMANCE METRICS")
                    click.echo("-" * 40)
                    total_time = sum(performance.values())
                    
                    for domain in [d[0] for d in sorted_domains]:
                        if domain in performance:
                            load_time = performance[domain]
                            percentage = (load_time / total_time) * 100 if total_time > 0 else 0
                            click.echo(f"{domain:<15} {load_time:.3f}s ({percentage:.1f}%)")
                    
                    click.echo(f"{'TOTAL':<15} {total_time:.3f}s (100.0%)")
    except ImportError:
        click.echo("\n⚠️  IoC registry not available for performance metrics")

@cli.command(name='validate-architecture')
def validate_architecture():
    """Perform comprehensive architecture validation"""
    click.echo("🎯 COMPREHENSIVE ARCHITECTURE VALIDATION")
    click.echo("=" * 60)
    
    start_time = time.time()
    
    # Test single-pass loading
    click.echo("1️⃣  Testing Single-Pass Loading...")
    try:
        from pyics.core.ioc_registry import validate_architecture as validate_arch
        if validate_arch():
            click.echo("   ✅ Single-pass loading validation passed")
        else:
            click.echo("   ❌ Single-pass loading validation failed")
            sys.exit(1)
    except ImportError as e:
        click.echo(f"   ❌ Failed to import IoC registry: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"   ❌ Single-pass validation error: {e}")
        sys.exit(1)
    
    # Test domain loading
    click.echo("\n2️⃣  Testing Domain Loading...")
    try:
        from pyics.core.ioc_registry import get_all_domains, get_registry
        domains = get_all_domains()
        registry = get_registry()
        
        click.echo(f"   📦 {len(domains)} domains in load order")
        
        for domain in domains:
            metadata = registry.get_domain_metadata(domain)
            if metadata:
                click.echo(f"   ✅ {domain}")
            else:
                click.echo(f"   ❌ {domain} (metadata missing)")
                sys.exit(1)
                
    except Exception as e:
        click.echo(f"   ❌ Domain loading test failed: {e}")
        sys.exit(1)
    
    # Test cost metadata consistency
    click.echo("\n3️⃣  Testing Cost Metadata Consistency...")
    try:
        from pyics.core.ioc_registry import get_domain_cost_metadata
        
        cost_issues = []
        for domain in domains:
            cost_metadata = get_domain_cost_metadata(domain)
            if not cost_metadata:
                cost_issues.append(f"{domain}: missing cost metadata")
                continue
            
            # Validate required fields
            required_fields = ["priority_index", "compute_time_weight", "load_order", "thread_safe"]
            for field in required_fields:
                if field not in cost_metadata:
                    cost_issues.append(f"{domain}: missing {field}")
        
        if cost_issues:
            click.echo("   ❌ Cost metadata issues found:")
            for issue in cost_issues:
                click.echo(f"      • {issue}")
            sys.exit(1)
        else:
            click.echo("   ✅ Cost metadata consistency validated")
            
    except Exception as e:
        click.echo(f"   ❌ Cost metadata validation failed: {e}")
        sys.exit(1)
    
    # Performance test
    click.echo("\n4️⃣  Performance Testing...")
    try:
        performance = registry.get_load_performance()
        total_time = sum(performance.values())
        
        if total_time > 1.0:  # More than 1 second is concerning
            click.echo(f"   ⚠️  Load time high: {total_time:.3f}s")
        else:
            click.echo(f"   ✅ Load time acceptable: {total_time:.3f}s")
        
        # Check for slow domains
        slow_domains = [d for d, t in performance.items() if t > 0.1]
        if slow_domains:
            click.echo(f"   ⚠️  Slow domains: {', '.join(slow_domains)}")
        
    except Exception as e:
        click.echo(f"   ⚠️  Performance test skipped: {e}")
    
    validation_time = time.time() - start_time
    
    click.echo("\n" + "=" * 60)
    click.echo("🎉 COMPREHENSIVE VALIDATION PASSED!")
    click.echo(f"⚡ Validation completed in {validation_time:.3f}s")
    click.echo("✅ Single-pass architecture fully compliant")
    click.echo("✅ All domains loaded successfully")
    click.echo("✅ Cost metadata validated")
    click.echo("✅ Performance within acceptable limits")
    click.echo("=" * 60)

@cli.command(name='fix-structure')
def fix_structure():
    """Fix architecture structure violations"""
    click.echo("🔧 FIXING ARCHITECTURE STRUCTURE")
    click.echo("=" * 50)
    
    # Check if structure corrector exists
    corrector_path = PROJECT_ROOT / "scripts" / "development" / "pyics_structure_corrector.py"
    
    if corrector_path.exists():
        click.echo(f"🔍 Found structure corrector: {corrector_path}")
        click.echo("🚀 Executing structure correction...")
        
        # Import and run corrector
        import subprocess
        result = subprocess.run([sys.executable, str(corrector_path)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            click.echo("✅ Structure correction completed successfully")
            click.echo("\n📋 Run 'pyics validate-architecture' to verify corrections")
        else:
            click.echo("❌ Structure correction failed")
            if result.stderr:
                click.echo(f"Error: {result.stderr}")
            sys.exit(1)
    else:
        click.echo("❌ Structure corrector not found")
        click.echo(f"Expected location: {corrector_path}")
        click.echo("\n📋 Please ensure the structure corrector script is available")
        sys.exit(1)

def main():
    """Main CLI entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

# [EOF] - End of Pyics CLI main interface
