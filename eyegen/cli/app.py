"""Typer app factory and command registration."""

import typer

from eyegen.cli import bonsai, config, coreml, generate, hf, models, status

app = typer.Typer(
    help=(
        "Generate images using MLX SD3.5, MFLUX (FLUX/FIBO/Z-Image), "
        "OllamaDiffuser GGUF, Bonsai (PrismML ternary), or CoreML "
        "(Apple Neural Engine) backends"
    ),
    no_args_is_help=True,
)

app.command()(generate.generate)
app.command()(models.pull)
app.command()(models.list_models)
app.command(name="save-model")(models.save_model_cmd)
app.command(name="clear-cache")(models.clear_cache)

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
