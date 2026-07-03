"""Model dropdown population mixin for MainWindow."""

from eyegen.config import Backend


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
        if backend == Backend.COREML:
            return self._get_coreml_models()
        elif backend == Backend.MFLUX:
            return self._get_mflux_models()
        elif backend == Backend.BONSAI:
            return self._get_bonsai_models()
        else:
            return self._get_auto_models()

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

    def _get_auto_models(self):
        try:
            from eyegen.config import MODELS_DIR

            if MODELS_DIR.is_dir():
                return [
                    (child.name, child.name)
                    for child in sorted(MODELS_DIR.iterdir())
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
