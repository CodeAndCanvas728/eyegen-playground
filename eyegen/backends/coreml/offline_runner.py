"""Run a converted CoreML bundle without the PyTorch model it came from.

Apple's ``python_coreml_stable_diffusion.pipeline`` starts with
``StableDiffusionPipeline.from_pretrained(model_version)``, so a bundle that
holds only the converted models cannot run. This script is executed by the
CoreML sidecar's Python: it builds the same pipeline from the bundle's
``.mlpackage``/``.mlmodelc`` files and the tokenizer/scheduler assets saved
next to them, then writes the image to the exact path it was given.

It deliberately imports nothing from EyeGen: the sidecar does not have the
application's dependencies, and importing ``eyegen`` would pull in Qt and MLX.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("eyegen.coreml.offline")

_COMPONENTS = ("text_encoder", "text_encoder_2", "unet", "vae_decoder", "safety_checker")
_COMPILED_NAMES = {
    "text_encoder": "TextEncoder",
    "text_encoder_2": "TextEncoder2",
    "unet": "Unet",
    "vae_decoder": "VAEDecoder",
    "safety_checker": "SafetyChecker",
}
_ASSETS = "pipeline_assets"
_COMPUTE_UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")


class BundleError(RuntimeError):
    """A bundle cannot be run as it is."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a converted CoreML Stable Diffusion bundle.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("-i", required=True, help="Directory holding the Core ML models.")
    parser.add_argument("-o", required=True, help="Image file to write.")
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument(
        "--width", type=int, default=None, help="Defaults to the size the models were exported at."
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--compute-unit", default="CPU_AND_NE", choices=_COMPUTE_UNITS)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--negative-prompt", default=None)
    return parser.parse_args(argv)


def _component_path(bundle: Path, name: str, sources: str):
    if sources == "packages":
        exact = bundle / f"{name}.mlpackage"
        if exact.is_dir():
            return exact
        matches = sorted(path for path in bundle.glob(f"*_{name}.mlpackage") if path.is_dir())
        return matches[0] if matches else None
    compiled = bundle / f"{_COMPILED_NAMES[name]}.mlmodelc"
    return compiled if compiled.is_dir() else None


def discover_components(bundle: Path):
    """The model files in *bundle*, and whether they are packages or compiled."""
    packages = [path for path in bundle.glob("*.mlpackage") if path.is_dir()]
    compiled = [path for path in bundle.glob("*.mlmodelc") if path.is_dir()]
    if packages and compiled:
        raise BundleError(f"{bundle} holds both .mlpackage and .mlmodelc models; use one kind.")
    if packages:
        sources = "packages"
    elif compiled:
        sources = "compiled"
    else:
        raise BundleError(f"{bundle} holds no .mlpackage or .mlmodelc models.")

    components = {name: _component_path(bundle, name, sources) for name in _COMPONENTS}
    missing = [name for name in ("text_encoder", "unet", "vae_decoder") if components[name] is None]
    if missing:
        raise BundleError(f"{bundle} is missing these model files: {', '.join(missing)}.")
    return components, sources


def has_models(bundle: Path) -> bool:
    """Whether *bundle* holds every model file the pipeline needs."""
    try:
        discover_components(bundle)
    except BundleError:
        return False
    return True


def check_assets(assets: Path) -> None:
    """Raise unless *assets* holds what a run loads besides the models.

    The conversion that writes these files checks them with this, so a bundle
    it declares finished is one this script will start.
    """
    tokenizer = assets / "tokenizer"
    if not (tokenizer / "vocab.json").is_file() or not (tokenizer / "merges.txt").is_file():
        raise BundleError("the CLIP tokenizer files (vocab.json and merges.txt) are missing")
    if not (assets / "scheduler" / "scheduler_config.json").is_file():
        raise BundleError("the scheduler config (scheduler_config.json) is missing")


def pipeline_assets_dir(bundle: Path) -> Path:
    """The tokenizer and scheduler *bundle* needs to run, or why it has none."""
    assets = bundle / _ASSETS
    try:
        check_assets(assets)
    except BundleError as exc:
        raise BundleError(
            f"{bundle} cannot run: {exc}. Re-run ./generate.py convert-coreml <hf-model> to "
            "add them; the models are not converted again."
        ) from exc
    return assets


def _metadata(bundle: Path) -> dict:
    try:
        value = json.loads((bundle / "model_index.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_tokenizer(path: Path):
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(str(path), local_files_only=True)


def _load_scheduler(path: Path):
    import diffusers

    config = json.loads((path / "scheduler_config.json").read_text())
    class_name = config.get("_class_name", "PNDMScheduler")
    scheduler_class = getattr(diffusers, class_name, None)
    if scheduler_class is None:
        raise BundleError(f"Unknown scheduler {class_name!r} in {path}.")
    return scheduler_class.from_pretrained(str(path))


def _load_feature_extractor(assets: Path):
    path = assets / "feature_extractor"
    if not (path / "preprocessor_config.json").is_file():
        return None
    from transformers import CLIPFeatureExtractor

    return CLIPFeatureExtractor.from_pretrained(str(path), local_files_only=True)


def build_pipeline(args):
    """Build Apple's CoreML pipeline from the bundle, without a diffusers model."""
    from python_coreml_stable_diffusion.coreml_model import CoreMLModel
    from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline

    bundle = Path(args.i).expanduser()
    components, sources = discover_components(bundle)
    assets = pipeline_assets_dir(bundle)
    meta = _metadata(bundle)
    xl = components["text_encoder_2"] is not None

    tokenizer = _load_tokenizer(assets / "tokenizer")
    tokenizer_2 = None
    if xl:
        if not (assets / "tokenizer_2" / "vocab.json").is_file():
            raise BundleError(f"{bundle} has an SDXL text encoder but no tokenizer_2.")
        tokenizer_2 = _load_tokenizer(assets / "tokenizer_2")

    feature_extractor = _load_feature_extractor(assets)
    safety_checker = None
    if components["safety_checker"] is not None:
        if feature_extractor is None:
            log.warning("No feature extractor in %s; disabling the safety checker.", bundle)
        else:
            safety_checker = CoreMLModel(
                str(components["safety_checker"]), args.compute_unit, sources=sources
            )

    def load(name):
        return CoreMLModel(str(components[name]), args.compute_unit, sources=sources)

    return CoreMLStableDiffusionPipeline(
        text_encoder=load("text_encoder"),
        text_encoder_2=load("text_encoder_2") if xl else None,
        unet=load("unet"),
        vae_decoder=load("vae_decoder"),
        scheduler=_load_scheduler(assets / "scheduler"),
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        controlnet=None,
        xl=xl,
        force_zeros_for_empty_prompt=bool(meta.get("force_zeros_for_empty_prompt", xl)),
        feature_extractor=feature_extractor,
        safety_checker=safety_checker,
    )


def _requested_size(pipe, args):
    """The size to generate, refusing one the bundle's models cannot make.

    The Core ML models are exported at one resolution, and the pipeline reads
    it off the unet's latent shape (``pipe.width``/``pipe.height``). Asking
    Apple's pipeline for another size only logs a warning and returns the
    native one, so a 1024 request against a 512 bundle would silently produce
    a 512 image.
    """
    native = (pipe.width, pipe.height)
    asked = (
        native[0] if args.width is None else args.width,
        native[1] if args.height is None else args.height,
    )
    if asked != native:
        raise BundleError(
            f"{args.i} generates {native[0]}x{native[1]} images only; "
            f"asked for {asked[0]}x{asked[1]}. Set --width and --height to "
            f"{native[0]} and {native[1]} to use it."
        )
    return asked


def run(args):
    """Generate the image and write it to the exact path in *args.o*."""
    import numpy as np

    if not 0 <= args.seed < 2**32:
        raise BundleError(f"Seed {args.seed} is outside the range 0 to 2**32-1.")
    # Apple's pipeline draws its latents from numpy's global generator
    # (`prepare_latents` calls np.random.randn), so this is what makes a run
    # reproducible. There is no generator or seed argument to hand it instead.
    np.random.seed(args.seed)
    pipe = build_pipeline(args)
    width, height = _requested_size(pipe, args)
    image = pipe(
        prompt=args.prompt,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
    )
    out_path = Path(args.o).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image["images"][0].save(out_path)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        run(args)
    except (BundleError, OSError, ValueError) as exc:
        # The wrapper reads this line back to tell the user what went wrong.
        log.error("Error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
