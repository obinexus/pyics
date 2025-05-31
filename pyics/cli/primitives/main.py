#!/usr/bin/env python3
"""
pyics/cli/primitives/main.py
CLI Interface for Primitives Domain

Engineering Lead: Nnamdi Okpala / OBINexus Computing
"""

import click
from pyics.core.primitives import get_domain_metadata, validate_configuration

@click.group()
def primitives_cli():
    """CLI commands for primitives domain"""
    pass

@primitives_cli.command()
def status():
    """Show domain status and configuration"""
    metadata = get_domain_metadata()
    click.echo(f"Domain: {metadata['domain_name']}")
    click.echo(f"Load Order: {metadata['cost_metadata']['load_order']}")
    click.echo(f"Valid: {validate_configuration()}")

if __name__ == "__main__":
    primitives_cli()
