"""Tests for the class MFLUX runs a model with.

mflux maps a name to the class that runs it. The table in ``eyegen/_mflux``
was this tool's own answer to that question and went stale: it was written
when mflux knew seven families, mflux now knows fourteen, and the table's
fall-through ran the newer ones as ``Flux1`` — which loads and renders
noise, so nothing said it was wrong.

These check the order now in force: mflux answers, and where it has no
answer the table decides what it decided before. The package is not
vendored here, so the mapping is stood in for, and the one test that
needs the real thing runs in a subprocess of its own, skipping where the
package is not installed.
"""

import json
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from eyegen._mflux import _resolve_mflux_class

_MODEL_PATHS = {
    "mflux.models.flux.variants.txt2img.flux": "Flux1",
    "mflux.models.flux2.variants.txt2img.flux2_klein": "Flux2Klein",
    "mflux.models.z_image.variants.z_image": "ZImage",
    "mflux.models.fibo.variants.txt2img.fibo": "FIBO",
    "mflux.models.qwen.variants.txt2img.qwen_image": "QwenImage",
    "mflux.models.seedvr2.variants.upscale.seedvr2": "SeedVR2",
    "mflux.models.flux3.variants.txt2img.flux3_klein": "Flux3Klein",
}


class ModelConfigError(ValueError):
    """What the real package raises on a name it cannot place."""


def _class(name: str):
    return type(name, (), {})


def _config(name: str, aliases=()):
    return SimpleNamespace(model_name=name, aliases=list(aliases))


def _put_mapping(monkeypatch, resolve_key, classes=None):
    """Put mflux's mapping where the import looks for it."""
    save = types.ModuleType("mflux.models.common.cli.save")
    save.MODEL_CLASSES = {} if classes is None else classes

    resolution = types.ModuleType("mflux.models.common.resolution.config_resolution")
    resolution.ConfigResolution = SimpleNamespace(resolve_key=resolve_key)

    exceptions = types.ModuleType("mflux.utils.exceptions")
    exceptions.ModelConfigError = ModelConfigError

    monkeypatch.setitem(sys.modules, "mflux.models.common.cli.save", save)
    monkeypatch.setitem(sys.modules, "mflux.models.common.resolution.config_resolution", resolution)
    monkeypatch.setitem(sys.modules, "mflux.utils.exceptions", exceptions)


def _put_model_classes(monkeypatch):
    """Where the table imports its classes from: one module, one name."""
    for path, class_name in _MODEL_PATHS.items():
        module = types.ModuleType(path)
        setattr(module, class_name, _class(class_name))
        monkeypatch.setitem(sys.modules, path, module)


def test_what_mflux_says_is_what_answers(monkeypatch):
    """The finding: krea-2 was run through Flux1, which cannot read it."""
    krea2 = _class("Krea2")
    _put_mapping(monkeypatch, resolve_key=lambda name: "krea-2", classes={"krea-2": krea2})

    resolved = _resolve_mflux_class(_config("krea/Krea-2-Turbo", aliases=["krea-2"]))

    assert resolved is krea2


def test_a_name_mflux_cannot_place_falls_back_to_the_table(monkeypatch):
    """Its rules raise on a name they have never seen; the table decides."""

    def cannot_place(name):
        raise ModelConfigError(f"Cannot infer base_model from {name}")

    _put_mapping(monkeypatch, resolve_key=cannot_place)
    _put_model_classes(monkeypatch)

    resolved = _resolve_mflux_class(_config("my/checkpoint", aliases=["flux2-klein-4b"]))

    assert resolved.__name__ == "Flux2Klein"


def test_a_key_without_a_class_falls_back_to_the_table(monkeypatch):
    """lens-turbo resolves to a key the package ships no class for."""
    _put_mapping(monkeypatch, resolve_key=lambda name: "lens-turbo", classes={})
    _put_model_classes(monkeypatch)

    resolved = _resolve_mflux_class(_config("Comfy-Org/Lens", aliases=["lens-turbo"]))

    assert resolved.__name__ == "Flux1"


def test_a_name_outside_the_registry_falls_back_to_the_table(monkeypatch):
    """A config the resolver built itself has no key to look up."""
    _put_mapping(monkeypatch, resolve_key=lambda name: None)
    _put_model_classes(monkeypatch)

    resolved = _resolve_mflux_class(_config("my/checkpoint", aliases=["z-image"]))

    assert resolved.__name__ == "ZImage"


def test_a_package_without_the_mapping_falls_back_to_the_table(monkeypatch):
    """A mflux older than the mapping: the import has nothing to give."""
    for path in (
        "mflux.models.common.cli.save",
        "mflux.models.common.resolution.config_resolution",
        "mflux.utils.exceptions",
    ):
        monkeypatch.setitem(sys.modules, path, types.ModuleType(path))
    _put_model_classes(monkeypatch)

    resolved = _resolve_mflux_class(_config("black-forest-labs/FLUX.1-dev"))

    assert resolved.__name__ == "Flux1"


_REPORT = '''
"""What the real mflux runs each of these with."""
import json
import sys
from types import SimpleNamespace

from eyegen._mflux import _resolve_mflux_class

try:
    import mflux.models.common.cli.save
except ImportError:
    print(json.dumps({"classes": None}))
else:
    models = json.loads(sys.argv[1])
    print(json.dumps({"classes": {
        model: _resolve_mflux_class(SimpleNamespace(model_name=model, aliases=[])).__name__
        for model in models
    }}))
'''

# The mapping the package answers with, as it stands in mflux 0.23.
_EXPECTED = {
    # Six of these were run through Flux1 before mflux was asked.
    "krea/Krea-2-Turbo": "Krea2",
    "ideogram-ai/ideogram-4-fp8": "Ideogram4",
    "baidu/ERNIE-Image": "ErnieImage",
    "Boogu/Boogu-Image-0.1-Turbo": "BooguImage",
    "Qwen/Qwen-Image-2.1": "QwenImage21Edit",
    "inclusionAI/Ming-Image-0.1-Design": "MingImage",
    "black-forest-labs/flux-3-image-klein": "Flux3Klein",
    "black-forest-labs/FLUX.2-klein-4B": "Flux2Klein",
    "Tongyi-MAI/Z-Image": "ZImage",
    "briaai/FIBO": "FIBO",
    "numz/SeedVR2_comfyUI": "SeedVR2",
    "black-forest-labs/FLUX.1-dev": "Flux1",
}

# Enough for mflux and mlx to import; they are heavy, and nothing here runs them.
_REPORT_TIMEOUT = 120


def test_mflux_runs_each_family_with_its_own_class():
    """The mapping, against the real package.

    In a subprocess rather than the suite's own interpreter: mflux loads
    mlx, a native extension, and another test module has already put mlx
    into ``sys.modules`` as a mock for the session — importing the real
    one there aborts the process.
    """
    reported = subprocess.run(  # noqa: S603 — a fixed script, run by this interpreter
        [sys.executable, "-c", _REPORT, json.dumps(list(_EXPECTED))],
        capture_output=True,
        text=True,
        timeout=_REPORT_TIMEOUT,
    )
    assert reported.returncode == 0, f"the report could not be written:\n{reported.stderr}"

    # The last line, because a library's warnings share the stream.
    classes = json.loads(reported.stdout.strip().splitlines()[-1])["classes"]
    if classes is None:
        pytest.skip("mflux with its own model mapping is not installed")

    assert classes == _EXPECTED
