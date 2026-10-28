"""Run a model through the stable-diffusion.cpp binary."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from eyegen.backends.runner import BaseSubprocessRunner
from eyegen.config import EyeGenConfig
from eyegen.model_specs import ModelSpec

from .constants import SDCLI_MODELS, SDCLI_ROLE_FLAGS
from .install import diffusion_path, get_sdcli_timeout, physical_memory, validate_sdcli_install
from .render import SdCliRenderMixin

log = logging.getLogger(__name__)


class SdCliWrapper(SdCliRenderMixin, BaseSubprocessRunner):
    """Adapter that runs a model through the ``sd-cli`` binary.

    The binary is a separate program rather than a Python library on
    purpose: FLUX.3 needs a build of stable-diffusion.cpp that understands
    the architecture, and the Python bindings on PyPI are older than the
    model. The command line is the interface that upstream documents, and it
    is the one that works on Metal.
    """

    # The flags this backend passes, and which of them take a value. Only
    # the ones `_build_command` emits: the guard is a check on what this
    # backend builds, not a list of what the binary could take.
    FLAGS_WITH_ARGS = {
        "--diffusion-model",
        "--high-noise-diffusion-model",
        "--uncond-diffusion-model",
        "--vae",
        "--llm",
        "--llm_vision",
        "--t5xxl",
        "--clip_l",
        "--clip_g",
        "--clip_vision",
        "--tokenizer",
        "--control-net",
        "-p",
        "--cfg-scale",
        "--steps",
        "--sampling-method",
        "-s",
        "-W",
        "-H",
        "-o",
        "-r",
        "-n",
    }
    FLAGS_WITHOUT_ARGS = {"--offload-to-cpu"}

    def __init__(self, config: EyeGenConfig, model: str, spec: Optional[ModelSpec] = None):
        super().__init__(config)
        if spec is None:
            if model not in SDCLI_MODELS:
                raise ValueError(f"Unknown stable-diffusion.cpp model: {model}")
            spec = ModelSpec.from_dict(model, SDCLI_MODELS[model])
        # The specification names the model: a model that is a path on this
        # machine has a name of its own, and it is the name that is said.
        #
        # A directory holding only the transformer is not the whole model:
        # sd-cli also needs the text encoder and the VAE. When the GGUF sat
        # in a HuggingFace cache layout, its repository is known and the
        # companions come from there.
        from .models import complete_local_spec

        completed = complete_local_spec(spec)
        from_repo = completed is not spec
        spec = completed
        self.model = spec.name
        self.spec = spec
        self._check_roles()

        status = validate_sdcli_install(self.model, spec=spec, config=config)
        if not status.installed and from_repo and not spec.is_local:
            # The companions are in a repository, not on this machine yet:
            # fetch them, then check again. A directory of the transformer
            # alone is the model before its companions arrived. A shipped
            # model is fetched on its own pull command, not behind a run.
            from .models import pull_sdcli_model

            # The fetch goes to the directory the validation just checked:
            # a bare call would resolve the root on its own (the env wins
            # over the config file there) and the companions would land in
            # a tree the next validation never looks at.
            fetched = (
                pull_sdcli_model(self.model, spec=spec, model_dir=status.model_dir) is not None
            )
            if fetched:
                status = validate_sdcli_install(self.model, spec=spec, config=config)
            if not status.installed:
                missing = ", ".join(sorted(set(status.missing_files)))
                raise RuntimeError(
                    f"The model at {diffusion_path(spec)} is only the transformer; "
                    f"sd-cli also needs its companions, which could not be fetched:\n{missing}\n"
                    f"The files come from {spec.repo} — run the generation again when "
                    "the download can complete, or fetch them with huggingface-cli."
                )
        if not status.installed:
            raise RuntimeError(status.message)
        self.binary = status.binary
        self.model_dir = status.model_dir
        self.timeout = get_sdcli_timeout()
        self._check_memory()

    def _check_memory(self) -> None:
        """Refuse a model too large for the machine before it loads.

        sd-cli maps the whole model into memory. A text encoder at full
        precision is tens of gigabytes; on a 16 GB machine that is a
        freeze or a reboot, not a slower render. The quantised file the
        picker prefers is the same model that fits.
        """
        total = 0
        for _name, source in self.spec.sources.items():
            try:
                if source.is_local:
                    total += Path(source.path).stat().st_size
                elif self.model_dir is not None:
                    total += (self.model_dir / source.name).stat().st_size
            except OSError:
                continue
        available = physical_memory()
        if total > available * 0.9:
            raise RuntimeError(
                f"The model at {diffusion_path(self.spec)} needs more memory than this "
                f"machine has: its files total {total / 1e9:.1f} GB, and the machine has "
                f"{available / 1e9:.0f} GB of RAM.\n"
                "sd-cli loads a model whole; with no room it is a system freeze. Use the "
                "quantised companion files (the picker already prefers them), or run on "
                "a machine with more memory."
            )

    def _check_roles(self) -> None:
        """Refuse a file whose role the binary has no flag for.

        A role the binary does not know would be dropped, and the model
        would run without it: a command line with no transformer is a
        failure the user sees much later, if at all.
        """
        unknown = [role for role in self.spec.sources if role not in SDCLI_ROLE_FLAGS]
        if unknown:
            known = ", ".join(sorted(SDCLI_ROLE_FLAGS))
            raise ValueError(
                f"The model '{self.model}' names a role the stable-diffusion.cpp "
                f"backend does not know: {', '.join(sorted(unknown))}. "
                f"The roles it knows are: {known}."
            )

    def _validate_cmd_args(self, cmd: List[str]) -> None:
        """Allow only the flags this backend passes."""
        i = 1
        while i < len(cmd):
            arg = cmd[i]
            if arg.startswith("-"):
                if arg in self.FLAGS_WITH_ARGS:
                    if i + 1 >= len(cmd):
                        raise ValueError(f"Flag {arg} requires a value argument.")
                    i += 2
                elif arg in self.FLAGS_WITHOUT_ARGS:
                    i += 1
                else:
                    raise ValueError(f"Unsafe subprocess flag detected: {arg}")
            else:
                i += 1

    def _path_for(self, source) -> str:
        """Where one file of the model is.

        A file that is already on this machine is passed where it is; a
        file that was downloaded is in the model's own directory.
        """
        if source.is_local:
            return str(Path(source.path).expanduser())
        return str(self.model_dir / source.name)

    def _build_command(
        self,
        prompt: str,
        output_path: Path,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: Optional[int],
        image_path: Optional[str],
        negative_prompt: str = "",
    ) -> List[str]:
        spec = self.spec
        cmd = [str(self.binary)]
        for role, flag in SDCLI_ROLE_FLAGS.items():
            source = spec.sources.get(role)
            if source is None:
                continue
            cmd += [flag, self._path_for(source)]
        cmd += [
            "-p",
            prompt,
            "--cfg-scale",
            str(guidance),
            "--steps",
            str(steps),
            "--sampling-method",
            spec.sampling_method or "euler",
            "-W",
            str(width),
            "-H",
            str(height),
            "-o",
            str(output_path),
        ]
        if self.config.sdcli_offload_to_cpu:
            cmd += ["--offload-to-cpu"]
        if seed is not None:
            cmd += ["-s", str(seed)]
        if image_path:
            cmd += ["-r", image_path]
        if negative_prompt:
            cmd += ["-n", negative_prompt]
        return cmd
