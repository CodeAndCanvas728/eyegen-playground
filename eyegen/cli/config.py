"""Config-related CLI commands."""

import json

import typer

from eyegen import CONFIG_FILE, load_config, save_config


def config_show():
    """Display current configuration."""
    config = load_config()
    typer.echo("📋 Current Configuration:")
    typer.echo(json.dumps(config, indent=2))


def config_set(
    key: str = typer.Argument(..., help="Config key (e.g., 'num_inference_steps')"),
    value: str = typer.Argument(..., help="Config value"),
):
    """Update a configuration value."""
    config = load_config()

    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    config[key] = parsed_value
    try:
        save_config(config)
        typer.echo(f"✓ Config saved to {CONFIG_FILE}")
        typer.echo(f"✓ Set {key} = {parsed_value}")
    except ValueError as val_err:
        typer.echo(f"❌ Configuration validation failed:\n   {val_err}", err=True)
        raise typer.Exit(1)
    except OSError as e:
        typer.echo(f"❌ Failed to save config: {e}", err=True)
        raise typer.Exit(1)


def config_reset():
    """Reset configuration to defaults."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
    typer.echo("✓ Configuration reset to defaults")
