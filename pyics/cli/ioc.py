#!/usr/bin/env python3
"""
pyics/cli/ioc.py
CLI IoC Registration and Discovery

Engineering Lead: Nnamdi Okpala / OBINexus Computing
"""

from typing import Dict, List
import importlib

REGISTERED_CLI_DOMAINS = ['transforms', 'protocols', 'routing', 'logic', 'validation', 'primitives', 'validators', 'structures', 'composition', 'safety', 'registry', 'transformations']

def discover_cli_commands() -> Dict[str, object]:
    """Discover all CLI command groups"""
    commands = {}
    
    for domain in REGISTERED_CLI_DOMAINS:
        try:
            module = importlib.import_module(f"pyics.cli.{domain}.main")
            cmd_name = f"{domain}_cli"
            if hasattr(module, cmd_name):
                commands[domain] = getattr(module, cmd_name)
        except ImportError:
            pass
    
    return commands

__all__ = ["REGISTERED_CLI_DOMAINS", "discover_cli_commands"]
