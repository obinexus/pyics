#!/usr/bin/env python3
"""
pyics/cli/validation/main.py
CLI Interface for Validation Domain

Engineering Lead: Nnamdi Okpala / OBINexus Computing
"""

import click
from pyics.core.validation import get_domain_metadata, validate_configuration

@click.group()
def validation_cli():
    """CLI commands for validation domain"""
    pass

@validation_cli.command()
def status():
    """Show domain status and configuration"""
    metadata = get_domain_metadata()
    click.echo(f"Domain: {metadata['domain_name']}")
    click.echo(f"Load Order: {metadata['cost_metadata']['load_order']}")
    click.echo(f"Valid: {validate_configuration()}")

if __name__ == "__main__":
    validation_cli()
