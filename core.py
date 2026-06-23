#!/usr/bin/env python3
"""Backward-compatibility shim: re-export the public EyeGen API.

New code should import from ``eyegen`` directly. This module is kept only
because ``core_bonsai_pipeline.py`` and ``core_coreml_pipeline.py`` still
import ``OUTPUT_DIR`` from here.
"""

from eyegen import *  # noqa: F401,F403
