#!/usr/bin/env python3
"""
pyics/cli/registry/main.py
CLI Interface for Registry Domain

Engineering Lead: Nnamdi Okpala / OBINexus Computing
"""

import click
from pyics.core.registry import get_domain_metadata, validate_configuration

@click.group()
def registry_cli():
    """CLI commands for registry domain"""
    pass

@registry_cli.command()
def status():
    """Show domain status and configuration"""
    metadata = get_domain_metadata()
    click.echo(f"Domain: {metadata['domain_name']}")
    click.echo(f"Load Order: {metadata['cost_metadata']['load_order']}")
    click.echo(f"Valid: {validate_configuration()}")

if __name__ == "__main__":
    registry_cli()
