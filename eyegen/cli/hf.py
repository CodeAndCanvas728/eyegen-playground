"""HuggingFace authentication CLI commands."""

from typing import Optional

import typer

from eyegen import hf_login, hf_logout, hf_status


def hf_login_cmd(
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        help="[DEPRECATED] HuggingFace access token (prompts securely if omitted)",
    ),
):
    """Log in to HuggingFace to access gated models."""
    if token is not None:
        typer.echo(
            "⚠️ WARNING: Passing the token via the command line option '--token' / '-t' "
            "is deprecated because it can leak to your shell history. "
            "For security, omit this option and enter the token at the prompt.",
            err=True,
        )
    else:
        token = typer.prompt("Enter your HuggingFace token", hide_input=True)

    try:
        info = hf_login(token)
        typer.echo(f"✅ Logged in as {info.get('name', 'unknown')}")
    except (OSError, ValueError) as e:
        typer.echo(f"❌ Login failed: {e}", err=True)
        raise typer.Exit(1) from e


def hf_status_cmd():
    """Check HuggingFace login status."""
    info = hf_status()
    if info:
        typer.echo(f"✅ Logged in as {info.get('name', 'unknown')}")
    else:
        typer.echo("Not logged in. Run hf-login to authenticate.")


def hf_logout_cmd():
    """Log out from HuggingFace."""
    try:
        hf_logout()
        typer.echo("✅ Logged out from HuggingFace")
    except (OSError, ValueError) as e:
        typer.echo(f"❌ Logout failed: {e}", err=True)
        raise typer.Exit(1) from e
