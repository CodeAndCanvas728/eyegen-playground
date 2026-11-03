"""Backend/controls visibility mixin for MainWindow."""

from eyegen import DEFAULT_CONFIG, resolve_model
from eyegen.backends.sdcli.constants import SDCLI_MODELS, family_settings
from eyegen.config import Backend
from eyegen.gui.state import settings_for_model


class MainWindowControlsMixin:
    # Why the last resolution failed, when it did. The hint shows it.
    _backend_problem = None

    def _on_mode_changed(self, index: int):
        is_img2img = index == 1
        self.img2img_controls.setVisible(is_img2img)
        self.width_combo.setEnabled(not is_img2img)
        self.height_combo.setEnabled(not is_img2img)
        self._update_backend_dependent_controls()

    def _on_backend_changed(self, _index: int):
        self._populate_model_dropdown()
        self._update_backend_dependent_controls()

    def _resolved_backend(self) -> Backend | None:
        """Return the concrete backend for the current model + dropdown.

        Returns ``None`` when the model name cannot be resolved to a backend
        (``resolve_model`` raises ``ValueError``). Callers must treat ``None``
        as "unresolved" — generation is blocked but the GUI stays alive.
        """
        override = self.backend_combo.currentData()
        model = self.model_input.text().strip() or DEFAULT_CONFIG.model
        import dataclasses

        config = dataclasses.replace(self.config, model=model)
        try:
            return resolve_model(model, override, config=config).backend
        except ValueError as exc:
            self._backend_problem = str(exc)
            return None

    def _update_backend_dependent_controls(self):
        """Show/hide/enable controls based on the resolved backend."""
        self._backend_problem = None
        backend = self._resolved_backend()
        self.generate_btn.setEnabled(backend is not None)
        if backend is None:
            self.backend_hint.setText(
                f"⚠ {self._backend_problem or 'Unrecognized model name for the selected backend.'} "
                "Pick a backend or enter a supported model to generate."
            )
            self.backend_hint.show()
            return
        is_mlx = backend == Backend.MLX
        is_mflux = backend == Backend.MFLUX
        is_ollama = backend == Backend.OLLAMA
        is_bonsai = backend == Backend.BONSAI
        is_coreml = backend == Backend.COREML

        self._update_t5_control(is_mlx, is_ollama, is_mflux, is_bonsai, is_coreml)

        is_img2img = self.mode_tabs.currentIndex() == 1
        self.img2img_warning.setVisible(is_img2img and is_mlx)

        if (is_bonsai or is_coreml) and is_img2img:
            self.mode_tabs.setCurrentIndex(0)
            is_img2img = False

        self.quantize_row.setVisible(is_mflux)
        self.model_path_row.setVisible(is_mflux)
        if is_mflux:
            self._refresh_model_path_status()
        else:
            self.model_input.setEnabled(True)
            self.model_input.setToolTip("Hugging Face model ID or OllamaDiffuser model name")
            self.quantize_combo.setEnabled(True)
            self.quantize_combo.setToolTip("MFLUX runtime quantization level")

        self.pull_btn.setVisible(is_ollama)

        self.bonsai_row.setVisible(is_bonsai)
        if is_bonsai:
            self._refresh_bonsai_status()
        self.coreml_row.setVisible(is_coreml)
        if is_coreml:
            self._refresh_coreml_status()

        self._update_backend_hint(backend, is_mflux, is_mlx, is_bonsai, is_coreml)
        self._apply_model_settings()
        self._apply_mflux_steps_guidance(is_mflux)

    def _update_t5_control(self, is_mlx, is_ollama, is_mflux, is_bonsai, is_coreml):
        self.t5_check.setEnabled(is_mlx)
        if is_ollama:
            tip = "T5 is not applicable for GGUF models"
        elif is_mflux:
            tip = "T5 is not applicable for MFLUX models"
        elif is_bonsai:
            tip = "T5 is not applicable for Bonsai (uses Qwen3-4B internally)"
        elif is_coreml:
            tip = "T5 is not applicable for CoreML SD 1.x/2.x models"
        else:
            tip = ""
        self.t5_check.setToolTip(tip)

    def _update_backend_hint(self, backend, is_mflux, is_mlx, is_bonsai, is_coreml):
        family = family_settings(self.model_input.text().strip())
        if is_mflux:
            hint = "💡 FLUX models: ~4 steps, 3.5–4.0 guidance."
            if not self.model_path_input.text().strip():
                hint += " Save a model locally for faster loading."
        elif is_mlx and self.model_input.text().strip() == "mlx-community/Lance-3B-AWQ-INT4":
            hint = (
                "💡 Lance-3B: Highly efficient multimodal model. "
                "Recommended size: 512x512 or 1024x1024."
            )
        elif is_bonsai:
            hint = (
                "💡 Bonsai: 4 steps, guidance 1.0, no CFG, no negative prompt, "
                "no img2img. Image dimensions must be multiples of 32."
            )
        elif is_coreml:
            hint = (
                "💡 CoreML: SD 1.x/2.x model. Image dimensions should be 512x512 "
                "and multiples of 8. First call pays CoreML compile cost; "
                "subsequent calls are fast."
            )
        elif backend == Backend.SDCLI and family is not None:
            hint = (
                f"💡 {family['name']}: {family['steps']} steps, "
                f"{family['guidance']} guidance, dimensions a multiple of 32."
            )
        else:
            self.backend_hint.hide()
            return
        self.backend_hint.setText(hint)
        self.backend_hint.show()

    def _apply_model_settings(self):
        """Show the steps and guidance the selected model wants.

        A model is not a window: the settings of the last model do not apply
        to the next. Each model keeps its own pair, and one with no pair of
        its own is shown what its family or its specification asks for.
        """
        if self._restoring_state:
            return
        model = self.model_input.text().strip() or DEFAULT_CONFIG.model
        if model == self._applied_model:
            return
        if self._applied_model:
            self._settings_by_model[self._applied_model] = {
                "steps": self.steps_spin.value(),
                "guidance": self.guidance_spin.value(),
            }
        self._applied_model = model
        steps, guidance = settings_for_model(
            self._settings_by_model, model, self._model_defaults(model)
        )
        self.steps_spin.setValue(steps)
        self.guidance_spin.setValue(guidance)

    def _model_defaults(self, model: str) -> tuple[int, float]:
        """The steps and guidance *model* wants before the user says otherwise."""
        import dataclasses

        try:
            config = dataclasses.replace(self.config, model=model)
            spec = resolve_model(model, self.backend_combo.currentData(), config=config).spec
        except (ValueError, ImportError, OSError):
            spec = None
        if spec is not None and spec.steps is not None:
            guidance = spec.guidance if spec.guidance is not None else DEFAULT_CONFIG.guidance_scale
            return spec.steps, guidance
        family = family_settings(model)
        if family is not None:
            return family["steps"], family["guidance"]
        shipped = SDCLI_MODELS.get(model)
        if shipped is not None and shipped.get("steps") is not None:
            steps = shipped["steps"]
            guidance = shipped.get("guidance")
            if guidance is None:
                guidance = DEFAULT_CONFIG.guidance_scale
            return steps, guidance
        return DEFAULT_CONFIG.num_inference_steps, DEFAULT_CONFIG.guidance_scale

    def _apply_mflux_steps_guidance(self, is_mflux: bool):
        """MFLUX models are distilled: four steps at a guidance of 4.0.

        The pair is remembered with the model like any other, so leaving
        MFLUX returns to the values the model before it had.
        """
        if self._restoring_state or not is_mflux:
            return
        self.steps_spin.setValue(4)
        self.guidance_spin.setValue(4.0)
