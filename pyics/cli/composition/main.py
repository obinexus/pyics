#!/usr/bin/env python3
"""
pyics/cli/composition/main.py
CLI Interface for Composition Domain

Engineering Lead: Nnamdi Okpala / OBINexus Computing
"""

import click
from pyics.core.composition import get_domain_metadata, validate_configuration

@click.group()
def composition_cli():
    """CLI commands for composition domain"""
    pass

@composition_cli.command()
def status():
    """Show domain status and configuration"""
    metadata = get_domain_metadata()
    click.echo(f"Domain: {metadata['domain_name']}")
    click.echo(f"Load Order: {metadata['cost_metadata']['load_order']}")
    click.echo(f"Valid: {validate_configuration()}")

if __name__ == "__main__":
    composition_cli()
