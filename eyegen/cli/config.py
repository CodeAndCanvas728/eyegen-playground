"""Config-related CLI commands."""

import json

import typer

from eyegen import CONFIG_FILE, EyeGenConfig, load_config, save_config


def config_show():
    """Display current configuration."""
    config = load_config()
    typer.echo("📋 Current Configuration:")
    typer.echo(json.dumps(config.to_dict(), indent=2))


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

    data = config.to_dict()
    data[key] = parsed_value
    try:
        save_config(EyeGenConfig.from_dict(data))
        typer.echo(f"✓ Config saved to {CONFIG_FILE}")
        typer.echo(f"✓ Set {key} = {parsed_value}")
    except ValueError as val_err:
        typer.echo(f"❌ Configuration validation failed:\n   {val_err}", err=True)
        raise typer.Exit(1) from val_err
    except OSError as e:
        typer.echo(f"❌ Failed to save config: {e}", err=True)
        raise typer.Exit(1) from e


def config_reset():
    """Reset configuration to defaults."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
    typer.echo("✓ Configuration reset to defaults")
