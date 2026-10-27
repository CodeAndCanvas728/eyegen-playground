"""The `pull` command: fetching a model, by name or from a repository."""

from typing import Annotated, Optional

import typer

from eyegen.config import Backend
from eyegen.model_catalogue import upsert_spec
from eyegen.model_specs import ModelSource, ModelSpec

# The roles a model can have, and the flag the stable-diffusion.cpp binary
# takes for each. The names are the binary's own.
_FILE_HELP = (
    "A file of the model, as role=repository:filename. The role is the part "
    "the file plays: diffusion_model, vae, llm, t5xxl, clip_l, clip_g, "
    "clip_vision, llm_vision, tokenizer, control_net. The repository may be "
    "left out when it is the one given by --from-hf."
)


def _parse_source(value: str, default_repo: Optional[str]) -> tuple[str, ModelSource]:
    """Read one ``role=repository:filename`` value."""
    if "=" not in value:
        raise ValueError(f"'{value}' must be written as role=repository:filename")
    role, _, source = value.partition("=")
    role = role.strip()
    if not role:
        raise ValueError(f"'{value}' names no role")
    if ":" in source:
        repo, _, filename = source.partition(":")
        repo = repo.strip()
    else:
        repo, filename = (default_repo or "").strip(), source.strip()
    if not repo:
        raise ValueError(
            f"the file '{filename}' has no repository: write it as role=repository:filename"
        )
    if not filename:
        raise ValueError(f"the file for the role '{role}' has no filename")
    return role, ModelSource(repo=repo, file=filename)


def _roles_for(backend: Backend) -> dict[str, str]:
    """The roles *backend* knows, and what it does with each."""
    if backend == Backend.SDCLI:
        from eyegen.backends.sdcli.constants import SDCLI_ROLE_FLAGS

        return SDCLI_ROLE_FLAGS
    if backend == Backend.OLLAMA:
        return {"diffusion_model": "the GGUF file of the model"}
    return {}


def _check_roles(roles: list[str], backend: Backend) -> None:
    """Refuse a role the backend does not know, and a role given twice.

    A role the backend does not know would be dropped, and the model would
    be fetched without a file it needs. A role given twice would silently
    keep the last one, and the first file would be fetched for nothing.
    """
    known = _roles_for(backend)
    seen: set[str] = set()
    for role in roles:
        if role not in known:
            names = ", ".join(sorted(known)) or "none"
            raise ValueError(
                f"the {backend.value} backend does not know the role '{role}'. "
                f"The roles it knows are: {names}."
            )
        if role in seen:
            raise ValueError(f"the role '{role}' is given twice")
        seen.add(role)


def build_spec(
    name: str,
    backend: str,
    repo: str,
    files: list[str],
    steps: Optional[int],
    guidance: Optional[float],
    sampler: Optional[str],
) -> ModelSpec:
    """The specification the command line describes."""
    try:
        backend_value = Backend(backend)
    except ValueError as exc:
        known = ", ".join(sorted(b.value for b in Backend))
        raise ValueError(f"'{backend}' is not a backend. Known backends: {known}.") from exc

    if backend_value in (Backend.MLX, Backend.MFLUX, Backend.BONSAI):
        raise ValueError(
            f"The {backend_value.value} backend runs a fixed set of models and cannot "
            f"fetch '{name}' from a repository. Name one of its own models instead."
        )
    if backend_value == Backend.COREML:
        raise ValueError(
            "A CoreML model is converted, not fetched: run "
            "./generate.py convert-coreml <hf-model>, then name the bundle it made."
        )

    parsed = [_parse_source(value, repo) for value in files]
    _check_roles([role for role, _ in parsed], backend_value)
    sources = {role: source for role, source in parsed}

    if backend_value == Backend.OLLAMA:
        if "diffusion_model" not in sources:
            raise ValueError(
                "an OllamaDiffuser model needs one GGUF file: "
                "--file diffusion_model=repository:filename"
            )
    elif not sources:
        raise ValueError(
            "a stable-diffusion.cpp model needs at least one file: "
            "--file diffusion_model=repository:filename"
        )

    return ModelSpec(
        name=name,
        backend=backend_value,
        sources=sources,
        steps=steps,
        guidance=guidance,
        sampling_method=sampler,
    )


def pull(
    model_name: Optional[str] = typer.Argument(
        None,
        help="Model name (e.g. flux.1-dev-gguf-q4ks, or flux-3-image-klein-gguf)",
    ),
    from_hf: Optional[str] = typer.Option(
        None,
        "--from-hf",
        help="A HuggingFace repository to take the model from",
    ),
    as_name: Optional[str] = typer.Option(
        None,
        "--as",
        help="The name to save the model under (default: the last part of the repository)",
    ),
    backend: str = typer.Option(
        "sdcli",
        "--backend",
        "-b",
        help="The backend that runs it: sdcli, ollamadiffuser",
    ),
    file: Annotated[Optional[list[str]], typer.Option("--file", help=_FILE_HELP)] = None,
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        help="The steps the model is trained for (a distilled model wants few)",
    ),
    guidance: Optional[float] = typer.Option(
        None,
        "--guidance",
        help="The guidance the model is trained for",
    ),
    sampler: Optional[str] = typer.Option(
        None,
        "--sampler",
        help="The sampler to use (euler, ddim, ...)",
    ),
):
    """Download a model.

    A GGUF model comes from OllamaDiffuser; a FLUX.3 model comes as three
    files — the transformer, the text encoder and the VAE — and is fetched
    from where each of them lives.

    A model the tool does not know comes from a repository: give the
    repository with --from-hf, the files with --file, and the model is
    written to your own catalogue and fetched.
    """
    if from_hf:
        _pull_from_hub(from_hf, as_name, backend, file or [], steps, guidance, sampler)
        return

    if not model_name:
        typer.echo("❌ Give a model name, or a repository with --from-hf", err=True)
        raise typer.Exit(1)

    typer.echo(f"📥 Pulling model: {model_name}")

    def on_progress(msg):
        if isinstance(msg, str):
            typer.echo(f"   {msg}")

    try:
        from eyegen import pull_any_model

        ok = pull_any_model(model_name, progress_callback=on_progress)
        if ok:
            typer.echo(f"✅ Model '{model_name}' is ready")
        else:
            typer.echo(f"❌ Failed to pull model '{model_name}'", err=True)
            raise typer.Exit(1)
    except (OSError, ValueError) as e:
        typer.echo(f"❌ Pull failed: {e}", err=True)
        raise typer.Exit(1) from e


def _pull_from_hub(
    repo: str,
    as_name: Optional[str],
    backend: str,
    files: list[str],
    steps: Optional[int],
    guidance: Optional[float],
    sampler: Optional[str],
) -> None:
    """Write the specification a repository describes, and fetch the model."""
    name = as_name or repo.split("/")[-1]
    try:
        spec = build_spec(name, backend, repo, files, steps, guidance, sampler)
    except ValueError as exc:
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(1) from exc

    try:
        path = upsert_spec(spec)
    except (OSError, ValueError) as exc:
        typer.echo(f"❌ Could not write the model catalogue: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"📝 Model '{name}' written to {path}")
    typer.echo(f"📥 Pulling model: {name}")

    def on_progress(msg):
        if isinstance(msg, str):
            typer.echo(f"   {msg}")

    try:
        from eyegen import pull_spec

        ok = pull_spec(spec, progress_callback=on_progress)
    except (OSError, ValueError, RuntimeError) as e:
        typer.echo(f"❌ Pull failed: {e}", err=True)
        raise typer.Exit(1) from e

    if not ok:
        typer.echo(f"❌ Failed to pull model '{name}'", err=True)
        raise typer.Exit(1)
    typer.echo(f"✅ Model '{name}' is ready")
    typer.echo(f"   Use it with: ./generate.py generate 'your prompt' --model {name}")
