"""Thin subprocess wrapper – replaces python_coreml_stable_diffusion.pipeline.

The upstream entry-point's ``main()`` unconditionally calls
``StableDiffusionPipeline.from_pretrained(model_version)`` which forces a
HuggingFace Hub fetch (or a local-cache lookup).  When the model has only
been converted to CoreML's ``.mlpackage`` format and is not present in the
HF cache, this always fails.

This module patches the intercepted module's ``main`` before any user code
(or frame-0 import side-effects) can call it, so the subprocess works
entirely offline.
"""

from __future__ import annotations

import importlib

# Resolve the location of the real offline runner relative to this file's
# package directory (eyegen/backends/coreml/).  Using a plain import avoids
# the need to pass a click or editable -e path to the subprocess.
_runner_mod_name = "eyegen.backends.coreml.offline_runner"


def _patched_main(argv=None):
    runner = importlib.import_module(_runner_mod_name)
    # Preserve the upstream CLI surface exactly: parse args then hand off.
    parser_cls = getattr(runner, "build_parser", None)
    if parser_cls is None:
        from argparse import ArgumentParser  # noqa: PLC0415

        build_parser = lambda: ArgumentParser()  # noqa: E731
    else:
        build_parser = parser_cls
    p = build_parser()
    args = p.parse_args(argv)
    main_fn = getattr(runner, "main", None)
    if main_fn is None:
        raise RuntimeError("offline_runner has no main()")
    return main_fn(args)


def _install_patch():
    # Replace main() in sys.modules[python_coreml_stable_diffusion.pipeline]
    # so any downstream code that imports that module gets the patched version.
    real_mod_name = "python_coreml_stable_diffusion.pipeline"
    real_mod = importlib.import_module(real_mod_name)
    real_mod.main = _patched_main  # type: ignore[attr-defined]
    return real_mod


if __name__ != "__main__":
    # We are being imported as -m – install the patch immediately.
    _install_patch()
else:
    # Stand-alone execution (python _offline_wrapper.py …)
    _patched_main()
