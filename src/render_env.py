from __future__ import annotations

import os


def configure_pyvista_env(*, offscreen: bool) -> None:
    """Configure PyVista/VTK before either library is imported."""
    if not offscreen:
        return

    defaults = {
        "PYVISTA_OFF_SCREEN": "true",
        "VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN": "1",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
