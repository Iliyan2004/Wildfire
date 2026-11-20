"""Backend choice: GPU (CUDA via cupy) first, CPU (numba, data in RAM) as the fallback."""
from __future__ import annotations

import time

from . import kernel_cpu, kernel_gpu


def simulate(grid, fuel_rows, tables, wind_ms, wind_from_deg, ignitions, hours, spot_rate=2e-4, seed=1,
             hour0=0, backend="auto", decay_min=0.0):
    """Returns (arrival, ros, fli, spots, info). arrival is minutes (inf = not reached)."""
    info = {}
    if backend in ("auto", "gpu") and kernel_gpu.available():
        try:
            t = time.time()
            out = kernel_gpu.simulate(grid, fuel_rows, tables, wind_ms, wind_from_deg, ignitions, hours,
                                      spot_rate=spot_rate, seed=seed, hour0=hour0, decay_min=decay_min)
            info.update(backend="gpu", solve_s=round(time.time() - t, 2), iterations=kernel_gpu.simulate.last_iterations)
            return (*out, info)
        except Exception as e:                       # e.g. CUDA unavailable or out of memory: fall back, say so
            if backend == "gpu":
                raise
            info["gpu_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    t = time.time()
    out = kernel_cpu.simulate(grid, fuel_rows, tables, wind_ms, wind_from_deg, ignitions, hours,
                              spot_rate=spot_rate, seed=seed, hour0=hour0, decay_min=decay_min)
    info.update(backend="cpu", solve_s=round(time.time() - t, 2))
    return (*out, info)


def simulate_batch(grid, fuel_rows, tables_list, wind_list, wdir_list, ignitions, hours, spot_rate=2e-4, seed=1, hour0=0, backend="auto", decay_min=0.0):
    """Ensemble members at once. GPU: one batched launch; otherwise the members run one after another on the CPU. Returns (arrival (M, ny, nx), info)."""
    import numpy as np
    info = {}
    if backend in ("auto", "gpu") and kernel_gpu.available():
        try:
            t = time.time()
            out = kernel_gpu.simulate_batch(grid, fuel_rows, tables_list, wind_list, wdir_list, ignitions, hours, spot_rate=spot_rate, seed=seed, hour0=hour0, decay_min=decay_min)
            info.update(backend="gpu-batch", solve_s=round(time.time() - t, 2), members=len(tables_list))
            return out, info
        except Exception as e:
            if backend == "gpu":
                raise
            info["gpu_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    t = time.time()
    outs = [kernel_cpu.simulate(grid, fuel_rows, tb, ws, wd, ignitions, hours, spot_rate=spot_rate, seed=seed + 100 + k, hour0=hour0, decay_min=decay_min)[0]
            for k, (tb, ws, wd) in enumerate(zip(tables_list, wind_list, wdir_list))]
    info.update(backend="cpu-loop", solve_s=round(time.time() - t, 2), members=len(tables_list))
    return np.stack(outs), info
