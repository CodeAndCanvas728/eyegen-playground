"""Typer app factory and command registration."""

import sys

import typer

from eyegen.cli import bonsai, config, coreml, generate, hf, models, pull, status, update


def _print_version() -> None:
    """Print the version of the installed package."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("eyegen")
    except PackageNotFoundError:
        installed = "unknown"
    typer.echo(f"eyegen {installed}")


app = typer.Typer(
    help=(
        "Generate images using MLX SD3.5, MFLUX (FLUX/FIBO/Z-Image), "
        "OllamaDiffuser GGUF, Bonsai (PrismML ternary), or CoreML "
        "(Apple Neural Engine) backends"
    ),
    no_args_is_help=True,
)

app.command()(generate.generate)
app.command()(pull.pull)
app.command()(models.list_models)
app.command(name="save-model")(models.save_model_cmd)
app.command(name="clear-cache")(models.clear_cache)
app.command(name="update-backends")(update.update_backends_cmd)

app.command(name="config-show")(config.config_show)
app.command(name="config-set")(config.config_set)
app.command(name="config-reset")(config.config_reset)

app.command(name="list-outputs")(status.list_outputs)
app.command()(status.status)

app.command(name="hf-login")(hf.hf_login_cmd)
app.command(name="hf-status")(hf.hf_status_cmd)
app.command(name="hf-logout")(hf.hf_logout_cmd)

app.command(name="setup-bonsai")(bonsai.setup_bonsai_cmd)
app.command(name="pull-bonsai")(bonsai.pull_bonsai_cmd)
app.command(name="list-bonsai-models")(bonsai.list_bonsai_models_cmd)

app.command(name="setup-coreml")(coreml.setup_coreml_cmd)
app.command(name="pull-coreml")(coreml.pull_coreml_cmd)
app.command(name="convert-coreml")(coreml.convert_coreml_cmd)
app.command(name="list-coreml-models")(coreml.list_coreml_models_cmd)


def main() -> None:
    """The console entry point.

    Typer resolves the command before it runs the group's callback, so
    "--version" is handled here rather than in a callback that would
    never be reached. It is the first argument only: a version after a
    command belongs to that command.
    """
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        _print_version()
        return
    app()
