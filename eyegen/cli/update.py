"""The `update-backends` CLI command."""

from __future__ import annotations

import typer

from eyegen.backends_update import apply_updates, format_report, survey


def update_backends_cmd(
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Install the updates, not just report them",
    ),
):
    """Report whether the backends are behind, and optionally update them.

    The backends are Python packages in the project venv, and a new model
    arrives when one of them is released. This asks PyPI what is out there
    and compares it with what is installed.
    """
    typer.echo(format_report(survey()))
    if not apply:
        typer.echo("\nRun this with --apply to install the updates.")
        return

    typer.echo("\nInstalling…")
    ok, output = apply_updates()
    if not ok:
        typer.echo(f"❌ The update failed: {output}", err=True)
        raise typer.Exit(1)
    typer.echo("✅ Updated.")
    typer.echo(format_report(survey()))
