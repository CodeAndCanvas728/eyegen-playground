"""Status and output-listing CLI commands."""

import importlib.util

import typer

from eyegen import (
    CONFIG_FILE,
    OUTPUT_DIR,
    PROJECT_ROOT,
    Backend,
    EyeGenConfig,
    detect_backend,
    hf_status,
    list_mflux_models,
    list_ollama_models,
    load_config,
)
from eyegen.backends import bonsai, coreml
from eyegen.config import pop_config_warnings


def list_outputs():
    """List all generated images."""
    images = sorted(OUTPUT_DIR.glob("*.png"))

    if not images:
        typer.echo("No images generated yet.")
        return

    typer.echo(f"📁 Generated images ({len(images)}):")
    for img in images:
        size = img.stat().st_size / 1024
        typer.echo(f"  • {img.name} ({size:.1f} KB)")


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version(name)
    except (OSError, ValueError, ImportError, AttributeError):
        return "unknown"


def _print_backend_status(config: EyeGenConfig):
    backend_setting = config.backend or Backend.AUTO.value
    try:
        resolved = detect_backend(config.model or "", backend_setting, config=config)
    except ValueError:
        resolved = backend_setting

    typer.echo("\n⚙️  Configuration:")
    typer.echo(f"  Model: {config.model or 'Not set'}")
    resolved_label = resolved.value if isinstance(resolved, Backend) else resolved
    typer.echo(f"  Backend: {backend_setting} → {resolved_label}")
    typer.echo(f"  Default steps: {config.num_inference_steps}")
    typer.echo(f"  Default guidance: {config.guidance_scale}")
    typer.echo(f"  Default size: {config.width}x{config.height}")


def _print_installed_packages(config: EyeGenConfig):
    _print_diffusionkit_status()
    _print_ollama_status()
    _print_mflux_status()
    _print_bonsai_status()
    _print_coreml_status()
    _print_output_status()
    _print_hf_status()


def _print_diffusionkit_status():
    if importlib.util.find_spec("diffusionkit") is not None:
        typer.echo(f"\n✅ diffusionkit: Installed (v{_package_version('diffusionkit')})")
    else:
        typer.echo("\n❌ diffusionkit: Not installed")


def _print_ollama_status():
    if importlib.util.find_spec("ollamadiffuser") is None:
        typer.echo("❌ ollamadiffuser: Not installed")
        return
    typer.echo(f"✅ ollamadiffuser: Installed (v{_package_version('ollamadiffuser')})")
    try:
        models = list_ollama_models()
        installed = models.get("installed", [])
        typer.echo(f"   GGUF models installed: {len(installed)}")
        for m in installed[:5]:
            typer.echo(f"     • {m}")
        if len(installed) > 5:
            typer.echo(f"     ... and {len(installed) - 5} more")
    except (OSError, ValueError, ImportError, AttributeError):
        pass


def _print_mflux_status():
    try:
        import mflux  # noqa: F401

        typer.echo("✅ mflux: Installed")
    except ImportError:
        typer.echo("❌ mflux: Not installed")
        return
    mflux_models = list_mflux_models()
    typer.echo(f"   Available models: {len(mflux_models)}")
    for m in mflux_models[:5]:
        typer.echo(f"     • {m['alias']}")
    if len(mflux_models) > 5:
        typer.echo(f"     ... and {len(mflux_models) - 5} more (run list-models to see all)")


def _print_bonsai_status():
    bonsai_status = bonsai.validate_bonsai_install()
    if bonsai_status.installed:
        bonsai_models = bonsai.list_bonsai_models()
        typer.echo("\n🌳 Bonsai (PrismML): Installed")
        typer.echo(f"   Vendor: {bonsai_status.bonsai_dir}")
        typer.echo(f"   Models: {len(bonsai_models)}")
        for m in bonsai_models:
            typer.echo(f"     • {m['alias']}")
    else:
        typer.echo("\n🌳 Bonsai (PrismML): Not installed (run setup-bonsai)")


def _print_coreml_status():
    coreml_status = coreml.validate_coreml_install()
    if coreml_status.installed:
        coreml_models = coreml.list_coreml_models()
        typer.echo(f"\n🍎 CoreML: Installed (venv={coreml_status.venv_python})")
        typer.echo(f"   Models: {len(coreml_models)}")
        for m in coreml_models:
            typer.echo(f"     • {m['name']}")
    else:
        typer.echo("\n🍎 CoreML: Not installed (run setup-coreml)")


def _print_output_status():
    images = list(OUTPUT_DIR.glob("*.png"))
    typer.echo(f"\n📊 Images generated: {len(images)}")


def _print_hf_status():
    info = hf_status()
    if info:
        typer.echo(f"🔑 HuggingFace: logged in as {info.get('name', 'unknown')}")
    else:
        typer.echo("🔑 HuggingFace: not logged in (run hf-login for gated models)")


def status():
    """Check system status and model availability."""
    typer.echo("🔍 MLX SD 3.5 Workspace Status")
    typer.echo(f"  Project root: {PROJECT_ROOT}")
    typer.echo(f"  Config file: {CONFIG_FILE}")
    typer.echo(f"  Output directory: {OUTPUT_DIR}")

    config = load_config()
    for w in pop_config_warnings():
        typer.echo(f"⚠️  {w}", err=True)
    _print_backend_status(config)
    _print_installed_packages(config)
