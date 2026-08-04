"""ctypes loading for the single Mojo graph-kernel compilation unit."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOIGRAPH_LIB") or os.path.join(ROOT, "dist", "libmojo-igraph.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mig_bfs": ([I] * 7, I),
    "mig_components": ([I] * 5, I),
    "mig_strong_components": ([I] * 9, I),
    "mig_pagerank": ([I, I, I, I, F, I, F, I, I], I),
    "mig_betweenness": ([I] * 12, None),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    """Build the shared library using the project task's canonical script."""
    if os.environ.get("MOJOIGRAPH_LIB"):
        if os.path.isfile(LIB):
            return LIB
        raise FileNotFoundError(f"MOJOIGRAPH_LIB does not name a shared library: {LIB}")
    source = os.path.join(ROOT, "src", "capi.mojo")
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    proc = subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")], cwd=ROOT,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(values: np.ndarray) -> int:
    return values.ctypes.data
