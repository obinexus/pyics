#!/usr/bin/env python3
"""
pyics/cli/protocols/main.py
CLI Interface for Protocols Domain

Engineering Lead: Nnamdi Okpala / OBINexus Computing
"""

import click
from pyics.core.protocols import get_domain_metadata, validate_configuration

@click.group()
def protocols_cli():
    """CLI commands for protocols domain"""
    pass

@protocols_cli.command()
def status():
    """Show domain status and configuration"""
    metadata = get_domain_metadata()
    click.echo(f"Domain: {metadata['domain_name']}")
    click.echo(f"Load Order: {metadata['cost_metadata']['load_order']}")
    click.echo(f"Valid: {validate_configuration()}")

if __name__ == "__main__":
    protocols_cli()
