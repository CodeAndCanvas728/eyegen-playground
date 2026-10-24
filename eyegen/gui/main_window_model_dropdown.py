"""Model dropdown population mixin for MainWindow."""

import logging

from eyegen.config import Backend

log = logging.getLogger("eyegen")


class MainWindowModelDropdownMixin:
    """Populates the model dropdown based on the selected backend."""

    def _on_model_dropdown_changed(self, _index: int):
        current_data = self.model_dropdown.currentData()
        if current_data is None:
            return
        if current_data == "__custom__":
            self.model_stack.setCurrentIndex(1)
            self.model_input.setFocus()
        else:
            self.model_input.setText(current_data)
            self._update_backend_dependent_controls()

    def _populate_model_dropdown(self):
        old_text = self.model_input.text()
        backend = self.backend_combo.currentData()

        self.model_dropdown.blockSignals(True)
        self.model_dropdown.clear()

        models = self._get_models_for_backend(backend)
        self._add_models_to_dropdown(models)
        self._add_custom_option()
        self._restore_previous_selection(old_text)

        self.model_dropdown.blockSignals(False)

    def _get_models_for_backend(self, backend):
        if backend == Backend.SDCLI:
            # The stable-diffusion.cpp listing already carries the user's own.
            return self._get_sdcli_models()
        own = self._get_own_models(backend)
        if backend == Backend.COREML:
            return own + self._get_coreml_models()
        elif backend == Backend.MFLUX:
            return own + self._get_mflux_models()
        elif backend == Backend.BONSAI:
            return own + self._get_bonsai_models()
        else:
            return self._get_auto_models()

    def _get_sdcli_models(self):
        try:
            from eyegen.backends.sdcli import list_sdcli_models

            return [(m["name"], m["name"]) for m in list_sdcli_models()]
        except (ImportError, OSError):
            return []

    def _get_coreml_models(self):
        try:
            from eyegen.backends.coreml import list_coreml_models

            return [(m["name"], m["name"]) for m in list_coreml_models()]
        except (ImportError, OSError):
            return []

    def _get_mflux_models(self):
        try:
            from eyegen._model_ops import list_mflux_models

            return [(f"{m['alias']} ({m['model_name']})", m["alias"]) for m in list_mflux_models()]
        except (ImportError, OSError):
            return []

    def _get_bonsai_models(self):
        try:
            from eyegen.backends.bonsai import list_bonsai_models

            return [(m["alias"], m["alias"]) for m in list_bonsai_models()]
        except (ImportError, OSError):
            return []

    def _get_own_models(self, backend=None):
        """The models in the user's own catalogue."""
        try:
            from eyegen.model_catalogue import load_catalogue

            specs = load_catalogue()
        except (ImportError, OSError) as exc:
            log.warning("Could not read the model catalogue: %s", exc)
            return []
        except ValueError as exc:
            # A catalogue that cannot be read is the user's problem to fix,
            # and it is worth saying so rather than showing an empty list.
            log.error("Could not read the model catalogue: %s", exc)
            return []
        return [
            (name, name)
            for name, spec in sorted(specs.items())
            if backend is None or spec.backend == backend
        ]

    def _get_auto_models(self):
        """Every model the tool can see, whatever backend runs it.

        A model on this machine is offered by path, and a catalogue often
        knows the same model by name. The local entry comes first and owns
        the display name, so the same model is not listed twice; nothing is
        lost, because the catalogue entry is still reachable by name.
        """
        models = (
            self._get_local_models()
            + self._get_own_models()
            + self._get_sdcli_models()
            + self._get_coreml_models()
            + self._get_saved_mflux_models()
        )
        seen = set()
        unique = []
        for display, value in models:
            if display in seen:
                continue
            seen.add(display)
            unique.append((display, value))
        return unique

    def _get_local_models(self):
        """The models on this machine that no catalogue describes.

        The value is the path: a name would have to be resolved back to this
        directory, and a catalogue may already answer to the name. A saved
        MFLUX model is left out here because it is offered by name above.
        """
        try:
            from eyegen.artifact_scan import scan_undescribed
            from eyegen.paths import models_dir

            return [
                (artifact.name, str(artifact.path))
                for artifact in scan_undescribed(models_dir())
                if artifact.backend != Backend.MFLUX
            ]
        except (ImportError, OSError):
            return []

    def _get_saved_mflux_models(self):
        """The models saved for a faster load."""
        try:
            from eyegen.paths import saved_mflux_dir

            directory = saved_mflux_dir()
            if directory.is_dir():
                return [
                    (child.name, child.name)
                    for child in sorted(directory.iterdir())
                    if child.is_dir() and not child.name.startswith(".")
                ]
            return []
        except (ImportError, OSError):
            return []

    def _add_models_to_dropdown(self, models):
        for display, value in models:
            self.model_dropdown.addItem(display, value)

    def _add_custom_option(self):
        self.model_dropdown.addItem("Custom\u2026", "__custom__")

    def _restore_previous_selection(self, old_text):
        idx = self.model_dropdown.findData(old_text)
        if idx >= 0:
            self.model_dropdown.setCurrentIndex(idx)
            self.model_stack.setCurrentIndex(0)
        else:
            for i in range(self.model_dropdown.count()):
                if self.model_dropdown.itemData(i) == old_text:
                    self.model_dropdown.setCurrentIndex(i)
                    self.model_stack.setCurrentIndex(0)
                    break
            else:
                self.model_input.setText(old_text)
                self.model_stack.setCurrentIndex(0)
