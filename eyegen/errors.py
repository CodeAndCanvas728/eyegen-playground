"""Custom exceptions used across EyeGen."""


class UnsupportedModelError(ValueError):
    """Raised when *detect_backend* cannot resolve a model name to any backend."""


class QuantizationError(RuntimeError):
    """Raised when MLX quantized-weight dequantization fails at inference.

    Callers can catch this to retry with ``quantize=None`` (full precision).
    """

    def __init__(self, original: Exception, model_name: str = ""):
        self.original = original
        self.model_name = model_name
        super().__init__(
            f"Quantization error for model '{model_name}': {original}. "
            "Try regenerating with full precision (quantize=None) or clear "
            "the model cache with: ./generate.py clear-cache"
        )


def _is_quantization_error(exc: Exception) -> bool:
    """Return True if *exc* looks like the MLX dequantize uint32 ValueError."""
    msg = str(exc).lower()
    return "dequantize" in msg or ("uint32" in msg and "matrix" in msg)
