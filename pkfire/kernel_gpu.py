"""CUDA solver for the minimum-travel-time fire front (same physics as kernel_cpu.py).

Dijkstra is sequential, so the GPU uses an exact parallel equivalent: the run is cut into one-hour bands
(the weather changes hourly). Within a band every cell whose arrival time is below the band end is "settled
enough" to spread: all cells pull candidate arrival times from their 16 neighbours (in place, min only) until
nothing changes. A cell with arrival < band end can only be improved by cells with an earlier arrival, which
are all in the iteration, so after convergence those arrivals are exact. Then the band's cells are finalised
(head rate/intensity written, ember spotting attempted with atomic-min) and the band is repeated if spotting
created new ignitions inside it. Embers use a counter-based RNG, so GPU spot fires are statistically but not
bit-for-bit equal to the CPU ones; with spotting off, GPU and CPU agree to float precision.
"""
from __future__ import annotations

import numpy as np

from . import fuels as fm

_SRC = r'''
#define PI_F 3.14159265f
__constant__ int DC[16] = {1,1,0,-1,-1,-1,0,1,2,2,1,-1,-2,-2,-1,1};
__constant__ int DR[16] = {0,1,1,1,0,-1,-1,-1,1,-1,2,2,-1,1,-2,-2};

struct P { const signed char* fuel; const float* slope; const float* upaz; const float* cont;
           int hour0, nh, nrow; const float *r0t,*irt,*Bt,*Ct,*Et,*rat,*sk,*waf,*tau; const float *wind,*windaz; const float* wmod; float tdec; int pad2; const float *cI0,*cCBD,*emc; const float* wrot; };

__device__ __forceinline__ void cellp(const P& p, int i, float t, float& ros, float& az, float& ecc, float& inten) {
    ros = 0.f; az = 0.f; ecc = 0.f; inten = 0.f;
    int row = p.fuel[i];
    if (row == 0) return;
    int h = p.hour0 + (int)(t / 60.0f);
    if (h >= p.nh) h = p.nh - 1;
    float r0 = p.r0t[h * p.nrow + row] * p.cont[i];
    if (p.tdec > 0.f) r0 = r0 / (1.f + t / p.tdec);
    if (r0 <= 1e-4f) return;
    float u = p.wind[h] * p.waf[row] * p.wmod[i] * 196.850394f;
    float phiw = (u > 0.f) ? p.Ct[row] * powf(u, p.Bt[row]) * powf(p.rat[row], -p.Et[row]) : 0.f;
    float tn = p.slope[i];
    float phis = p.sk[row] * tn * tn;
    float dw = r0 * phiw, ds = r0 * phis;
    float vx = dw * sinf(p.windaz[h] + p.wrot[i]) + ds * sinf(p.upaz[i]);
    float vy = dw * cosf(p.windaz[h] + p.wrot[i]) + ds * cosf(p.upaz[i]);
    float mag = sqrtf(vx * vx + vy * vy);
    az = (mag > 1e-9f) ? atan2f(vx, vy) : p.windaz[h] + p.wrot[i];
    float phie = mag / r0, ueff = 0.f;
    if (phie > 0.f) {
        ueff = powf(phie * powf(p.rat[row], p.Et[row]) / p.Ct[row], 1.0f / p.Bt[row]);
        float cap = 0.9f * p.irt[h * p.nrow + row];
        if (ueff > cap) { ueff = cap; phie = p.Ct[row] * powf(cap, p.Bt[row]) * powf(p.rat[row], -p.Et[row]); }
    }
    ros = r0 * (1.f + phie);
    if (ros > 200.f) ros = 200.f;
    float ums = ueff / 196.850394f;                     // effective wind in m/s (Anderson's coefficients are per m/s)
    float lb = 0.936f * expf(0.2566f * ums) + 0.461f * expf(-0.1548f * ums) - 0.397f;
    lb = fminf(fmaxf(lb, 1.f), 8.f);
    ecc = sqrtf(1.f - 1.f / (lb * lb));
    inten = p.irt[h * p.nrow + row] * 0.18927f * ros * p.tau[row];
    if (p.cCBD[i] > 0.f && inten >= p.cI0[i]) {                      // crown fire (Van Wagner initiation, Cruz active rate)
        float ros_c = 11.02f * powf(p.wind[h] * 3.6f, 0.9f) * powf(p.cCBD[i], 0.19f) * expf(-0.17f * p.emc[h]);
        if (ros_c >= 3.f / p.cCBD[i]) {
            if (ros_c > ros) ros = fminf(ros_c, 200.f);
            inten = inten + 150.f * ros_c;
        }
    }
}

extern "C" __global__ void relax(P p, int ny, int nx, float dx, float* arr, float band_end, int* changed,
                                 int r0, int c0, int rh, int cw) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= rh * cw) return;
    int r = r0 + q / cw, c = c0 + q % cw;
    int j = r * nx + c;
    if (p.fuel[j] == 0) return;
    float cur = arr[j], best = cur;
    for (int m = 0; m < 16; m++) {
        int cc = c - DC[m], rr = r - DR[m];
        if (cc < 0 || cc >= nx || rr < 0 || rr >= ny) continue;
        int i = rr * nx + cc;
        float ti = arr[i];
        if (!(ti < band_end)) continue;
        float ros, az, ecc, inten;
        cellp(p, i, ti, ros, az, ecc, inten);
        if (ros <= 0.f) continue;
        float stepaz = atan2f((float)DC[m], (float)(-DR[m]));
        float rate = ros * (1.f - ecc) / (1.f - ecc * cosf(stepaz - az));
        if (rate < 1e-3f) continue;
        float tj = ti + dx * sqrtf((float)(DC[m] * DC[m] + DR[m] * DR[m])) / rate;
        if (tj < best) best = tj;
    }
    if (best < cur) { arr[j] = best; *changed = 1; }
}

__device__ __forceinline__ float max_spot_m(float hfi, float u, float A, float B) {
    if (A <= 0.f || hfi <= 0.f) return 0.f;
    u = fmaxf(u, 0.1f);
    float E = hfi * A * powf(u, B), H = 0.173f * sqrtf(E);
    if (H <= 0.f) return 0.f;
    float h = powf(H, 0.337f) - 1.22f;
    if (!(h > 0.1f && H > h * 1.01f)) return 0.f;
    float u_h = u * powf(fmaxf(h, 6.f) / 10.f, 1.f / 7.f), u_H = u * powf(H / 10.f, 1.f / 7.f);
    float descent = 1.3f * 3.6f * u_h * sqrtf(h) * (0.362f + sqrtf(H / h) * 0.5f * logf(H / h));
    float lofting = (h < 10.f) ? 2.78f * u_H * sqrtf(H) : u_h * sqrtf(H) * (2.13f + logf(H / h));
    return fminf(fmaxf(descent + lofting, 0.f), 2000.f);
}

__device__ __forceinline__ unsigned hsh(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
__device__ __forceinline__ float u01(unsigned s) { return ((hsh(s) >> 8) + 0.5f) * (1.0f / 16777216.0f); }

extern "C" __global__ void finalize(P p, int ny, int nx, float dx, float* arr, float band_end, unsigned char* fin,
        float* ros_out, float* fli_out, const float* spotA, const float* spotB, const float* spot_ign, float spot_rate, unsigned seed, float tmax,
        float* spots, int* nspots, int max_spots, int* newspots, int r0, int c0, int rh, int cw) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= rh * cw) return;
    int i = (r0 + q / cw) * nx + (c0 + q % cw);
    if (fin[i]) return;
    float t = arr[i];
    if (!(t < band_end) || p.fuel[i] == 0) return;
    fin[i] = 1;
    float ros, az, ecc, inten;
    cellp(p, i, t, ros, az, ecc, inten);
    ros_out[i] = ros; fli_out[i] = inten;
    int row = p.fuel[i];
    if (ros <= 0.f || spot_rate <= 0.f || spotA[row] <= 0.f || inten <= 100.f) return;
    unsigned s = (unsigned)i * 2654435761u ^ (seed * 0x9E3779B9u);
    if (u01(s + 1) >= spot_rate * fminf(inten / 1000.f, 1.f)) return;
    int h = p.hour0 + (int)(t / 60.f);
    if (h >= p.nh) h = p.nh - 1;
    float u10 = p.wind[h];
    float dmax = max_spot_m(inten, u10, spotA[row], spotB[row]);
    if (dmax <= 30.f) return;
    float g1 = sqrtf(-2.f * logf(u01(s + 2))) * cosf(2.f * PI_F * u01(s + 3));
    float g2 = sqrtf(-2.f * logf(u01(s + 2))) * sinf(2.f * PI_F * u01(s + 3));
    float d = fminf(dmax, dmax * 0.35f * expf(0.6f * g1));
    float ang = p.windaz[h] + g2 * 0.17f;
    int r = i / nx, c = i - r * nx;
    int tc = c + (int)rintf(d * sinf(ang) / dx), tr = r - (int)rintf(d * cosf(ang) / dx);
    if (tc < 0 || tc >= nx || tr < 0 || tr >= ny) return;
    if (u01(s + 4) >= spot_ign[h]) return;
    int j = tr * nx + tc;
    if (p.fuel[j] == 0) return;
    float tj = t + 3.f + d / (fmaxf(u10, 1.f) * 60.f);
    if (tj <= tmax && tj < arr[j]) {
        atomicMin((int*)&arr[j], __float_as_int(tj));          // positive floats order like ints
        int k = atomicAdd(nspots, 1);
        if (k < max_spots) { spots[3 * k] = (float)i; spots[3 * k + 1] = (float)j; spots[3 * k + 2] = tj; }
        atomicAdd(newspots, 1);
    }
}
'''

_mod = None


def _module():
    global _mod
    if _mod is None:
        import cupy as cp
        _mod = cp.RawModule(code=_SRC, options=("--std=c++14",), name_expressions=None)
    return _mod


def available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def simulate(grid, fuel_rows, tables, wind_ms, wind_from_deg, ignitions, hours, spot_rate=2e-4, seed=1,
             hour0=0, max_spots=20000, decay_min=0.0):
    import cupy as cp
    ny, nx = grid["ny"], grid["nx"]
    n = ny * nx
    f32 = lambda a: cp.asarray(np.ascontiguousarray(a, np.float32))
    windaz = np.deg2rad((np.asarray(wind_from_deg, np.float64) + 180.0) % 360.0)
    dev = dict(
        fuel=cp.asarray(np.ascontiguousarray(fuel_rows.ravel(), np.int8)), slope=f32(grid["slope"].ravel()),
        upaz=f32(grid["upaz"].ravel()), cont=f32(grid["cont"].ravel()), r0t=f32(tables["r0"].ravel()),
        irt=f32(tables["ir"].ravel()), Bt=f32(tables["B"]), Ct=f32(tables["C"]), Et=f32(tables["E"]),
        rat=f32(tables["ratio"]), sk=f32(tables["slope_k"]), waf=f32(tables["waf"]), tau=f32(tables["tau"]),
        wrot=f32(grid["wrot"].ravel() if "wrot" in grid else np.zeros(n, np.float32)), cI0=f32(fm.crown_cells(grid, tables, fuel_rows)[0]), cCBD=f32(fm.crown_cells(grid, tables, fuel_rows)[1]), emc=f32(tables["emc_pct"]), wmod=f32(grid["wmod"].ravel() if "wmod" in grid else np.ones(n, np.float32)), wind=f32(wind_ms), windaz=f32(windaz), spotA=f32(tables["spot_a"]), spotB=f32(tables["spot_b"]), spotign=f32(tables["spot_ign"]))
    nh, nrow = len(wind_ms), tables["r0"].shape[1]
    # kernel parameter struct: pointers then ints, matching `struct P` (8-byte pointers first would break
    # alignment, so build it with a numpy structured dtype that mirrors the C layout)
    ptr = lambda k: dev[k].data.ptr
    P = np.zeros((), dtype=np.dtype([
        ("fuel", "u8"), ("slope", "u8"), ("upaz", "u8"), ("cont", "u8"),
        ("hour0", "i4"), ("nh", "i4"), ("nrow", "i4"), ("pad", "i4"),
        ("r0t", "u8"), ("irt", "u8"), ("Bt", "u8"), ("Ct", "u8"), ("Et", "u8"), ("rat", "u8"), ("sk", "u8"),
        ("waf", "u8"), ("tau", "u8"), ("wind", "u8"), ("windaz", "u8"), ("wmod", "u8"), ("tdec", "f4"), ("pad2", "i4"), ("cI0", "u8"), ("cCBD", "u8"), ("emc", "u8"), ("wrot", "u8")], align=False))
    for k in ("fuel", "slope", "upaz", "cont", "r0t", "irt", "Bt", "Ct", "Et", "rat", "sk", "waf", "tau", "wind", "windaz", "wmod", "cI0", "cCBD", "emc", "wrot"):
        P[k] = ptr(k)
    P["hour0"], P["nh"], P["nrow"] = hour0, nh, nrow
    P["tdec"] = decay_min

    mod = _module()
    relax, finalize = mod.get_function("relax"), mod.get_function("finalize")
    arr = cp.full(n, np.inf, cp.float32)
    ros_o, fli_o = cp.zeros(n, cp.float32), cp.zeros(n, cp.float32)
    fin = cp.zeros(n, cp.uint8)
    spots = cp.zeros(3 * max_spots, cp.float32)
    nsp, newsp, changed = cp.zeros(1, cp.int32), cp.zeros(1, cp.int32), cp.zeros(1, cp.int32)
    fuel_h = np.asarray(fuel_rows).ravel()
    for r, c, t0 in ignitions:
        i = r * nx + c
        if fuel_h[i] != 0:
            arr[i] = min(float(arr[i]), float(t0))
    tmax = float(hours * 60.0)
    threads = 256
    margin = int(min(200.0 * 60.0 / float(grid["dx"]) + 4, max(ny, nx)))   # fastest possible head advance in one band
    dx = np.float32(grid["dx"])
    Pk = np.void(P)
    nbands = int(np.ceil(tmax / 60.0))
    iters = 0
    for b in range(nbands):
        band_end = np.float32(min(60.0 * (b + 1), tmax + 1e-3))
        a2 = arr.reshape(ny, nx) < band_end
        rows, cols = cp.flatnonzero(a2.any(axis=1)), cp.flatnonzero(a2.any(axis=0))
        if rows.size == 0:
            continue
        r0 = max(int(rows[0]) - margin, 0); r1 = min(int(rows[-1]) + margin + 1, ny)
        c0 = max(int(cols[0]) - margin, 0); c1 = min(int(cols[-1]) + margin + 1, nx)
        rh, cw = r1 - r0, c1 - c0
        blocks = (rh * cw + threads - 1) // threads
        reg = (np.int32(r0), np.int32(c0), np.int32(rh), np.int32(cw))
        while True:
            while True:                                           # relax to convergence
                changed[0] = 0
                for _ in range(6):
                    relax((blocks,), (threads,), (P, np.int32(ny), np.int32(nx), dx, arr, band_end, changed) + reg)
                iters += 6
                if int(changed[0]) == 0:
                    break
            newsp[0] = 0
            finalize((blocks,), (threads,), (P, np.int32(ny), np.int32(nx), dx, arr, band_end, fin, ros_o, fli_o,
                                             dev["spotA"], dev["spotB"], dev["spotign"], np.float32(spot_rate), np.uint32(seed), np.float32(tmax),
                                             spots, nsp, np.int32(max_spots), newsp) + reg)
            if int(newsp[0]) == 0:
                break
    cp.cuda.Stream.null.synchronize()
    ns = min(int(nsp[0]), max_spots)
    sp = spots[:3 * ns].reshape(ns, 3).get().astype(np.float64)
    out = (arr.get().reshape(ny, nx), ros_o.get().reshape(ny, nx), fli_o.get().reshape(ny, nx), sp)
    simulate.last_iterations = iters
    return out


# ====================================================================================================================
# Batched solver: many members (ensemble) in one launch. The kernels are derived from the single-run source above by
# giving every per-hour weather array a leading member index, so that the physics is identical by construction.
# ====================================================================================================================
def _derive_batch(src):
    def rep(a, b, n=None):
        nonlocal src
        c = src.count(a)
        assert c >= 1 and (n is None or c == n), (a, c)
        src = src.replace(a, b)
    rep("cellp(const P& p, int i, float t,", "cellp(const P& p, int mem, int i, float t,", 1)
    rep("p.r0t[h * p.nrow + row]", "p.r0t[((long)mem * p.nh + h) * p.nrow + row]", 1)
    rep("p.irt[h * p.nrow + row]", "p.irt[((long)mem * p.nh + h) * p.nrow + row]", 2)
    rep("p.wind[h]", "p.wind[(long)mem * p.nh + h]")
    rep("p.windaz[h]", "p.windaz[(long)mem * p.nh + h]")
    rep("p.emc[h]", "p.emc[(long)mem * p.nh + h]", 1)
    rep("spot_ign[h]", "spot_ign[(long)mem * p.nh + h]", 1)
    rep("float* arr, float band_end, int* changed,", "float* arr0, float band_end, int* changed,", 1)
    rep("    if (q >= rh * cw) return;\n    int r = r0 + q / cw, c = c0 + q % cw;", "    if (q >= rh * cw) return;\n    int mem = blockIdx.y;\n    float* arr = arr0 + (long)mem * ((long)ny * nx);\n    int r = r0 + q / cw, c = c0 + q % cw;", 1)
    rep("        cellp(p, i, ti, ros, az, ecc, inten);", "        cellp(p, mem, i, ti, ros, az, ecc, inten);", 1)
    rep("float* arr, float band_end, unsigned char* fin,", "float* arr0, float band_end, unsigned char* fin0,", 1)
    rep("    if (q >= rh * cw) return;\n    int i = (r0 + q / cw) * nx + (c0 + q % cw);\n    if (fin[i]) return;", "    if (q >= rh * cw) return;\n    int mem = blockIdx.y;\n    float* arr = arr0 + (long)mem * ((long)ny * nx);\n    unsigned char* fin = fin0 + (long)mem * ((long)ny * nx);\n    int i = (r0 + q / cw) * nx + (c0 + q % cw);\n    if (fin[i]) return;", 1)
    rep("    cellp(p, i, t, ros, az, ecc, inten);\n    ros_out[i] = ros; fli_out[i] = inten;", "    cellp(p, mem, i, t, ros, az, ecc, inten);", 1)
    rep("(seed * 0x9E3779B9u)", "((seed + (unsigned)mem) * 0x9E3779B9u)", 1)
    return src


_modb = None


def simulate_batch(grid, fuel_rows, tables_list, wind_list, wdir_list, ignitions, hours, spot_rate=2e-4, seed=1, hour0=0, decay_min=0.0):
    """Runs len(tables_list) members at once. Members share terrain, fuel and ignitions; they differ in weather and in the hourly rate
    tables (moisture). Returns arrival times (M, ny, nx) in minutes (inf = not reached), float32."""
    import cupy as cp
    global _modb
    M = len(tables_list)
    ny, nx = grid["ny"], grid["nx"]
    n = ny * nx
    f32 = lambda a: cp.asarray(np.ascontiguousarray(a, np.float32))
    stack = lambda key: f32(np.stack([t[key] for t in tables_list]))
    windaz = np.stack([np.deg2rad((np.asarray(wd, np.float64) + 180.0) % 360.0) for wd in wdir_list])
    t0 = tables_list[0]
    nh, nrow = len(wind_list[0]), t0["r0"].shape[1]
    dev = dict(fuel=cp.asarray(np.ascontiguousarray(fuel_rows.ravel(), np.int8)), slope=f32(grid["slope"].ravel()), upaz=f32(grid["upaz"].ravel()),
               cont=f32(grid["cont"].ravel()), r0t=stack("r0"), irt=stack("ir"), Bt=f32(t0["B"]), Ct=f32(t0["C"]), Et=f32(t0["E"]), rat=f32(t0["ratio"]),
               sk=f32(t0["slope_k"]), waf=f32(t0["waf"]), tau=f32(t0["tau"]), wind=f32(np.stack([np.asarray(w, np.float64) for w in wind_list])), windaz=f32(windaz),
               wmod=f32(grid["wmod"].ravel() if "wmod" in grid else np.ones(n, np.float32)), wrot=f32(grid["wrot"].ravel() if "wrot" in grid else np.zeros(n, np.float32)),
               cI0=f32(fm.crown_cells(grid, t0, fuel_rows)[0]), cCBD=f32(fm.crown_cells(grid, t0, fuel_rows)[1]), emc=stack("emc_pct"), spotA=f32(t0["spot_a"]), spotB=f32(t0["spot_b"]), spotign=stack("spot_ign"))
    ptr = lambda k: dev[k].data.ptr
    P = np.zeros((), dtype=np.dtype([("fuel", "u8"), ("slope", "u8"), ("upaz", "u8"), ("cont", "u8"), ("hour0", "i4"), ("nh", "i4"), ("nrow", "i4"), ("pad", "i4"),
                                      ("r0t", "u8"), ("irt", "u8"), ("Bt", "u8"), ("Ct", "u8"), ("Et", "u8"), ("rat", "u8"), ("sk", "u8"), ("waf", "u8"), ("tau", "u8"),
                                      ("wind", "u8"), ("windaz", "u8"), ("wmod", "u8"), ("tdec", "f4"), ("pad2", "i4"), ("cI0", "u8"), ("cCBD", "u8"), ("emc", "u8"), ("wrot", "u8")], align=False))
    for k in ("fuel", "slope", "upaz", "cont", "r0t", "irt", "Bt", "Ct", "Et", "rat", "sk", "waf", "tau", "wind", "windaz", "wmod", "cI0", "cCBD", "emc", "wrot"):
        P[k] = ptr(k)
    P["hour0"], P["nh"], P["nrow"], P["tdec"] = hour0, nh, nrow, decay_min
    if _modb is None:
        _modb = cp.RawModule(code=_derive_batch(_SRC), options=("--std=c++14",))
    relax, finalize = _modb.get_function("relax"), _modb.get_function("finalize")
    arr = cp.full(M * n, np.inf, cp.float32)
    fin = cp.zeros(M * n, cp.uint8)
    ros_o, fli_o = cp.zeros(1, cp.float32), cp.zeros(1, cp.float32)
    spots = cp.zeros(3 * 16, cp.float32)
    nsp, newsp, changed = cp.zeros(1, cp.int32), cp.zeros(1, cp.int32), cp.zeros(1, cp.int32)
    fuel_h = np.asarray(fuel_rows).ravel()
    for m in range(M):
        for r, c, t0m in ignitions:
            i = r * nx + c
            if fuel_h[i] != 0:
                arr[m * n + i] = min(float(arr[m * n + i]), float(t0m))
    tmax = float(hours * 60.0)
    threads = 256
    margin = int(min(200.0 * 60.0 / float(grid["dx"]) + 4, max(ny, nx)))
    dx = np.float32(grid["dx"])
    Pk = np.void(P)
    for b in range(int(np.ceil(tmax / 60.0))):
        band_end = np.float32(min(60.0 * (b + 1), tmax + 1e-3))
        a3 = arr.reshape(M, ny, nx) < band_end
        a2 = a3.any(axis=0)
        rows, cols = cp.flatnonzero(a2.any(axis=1)), cp.flatnonzero(a2.any(axis=0))
        if rows.size == 0:
            continue
        r0 = max(int(rows[0]) - margin, 0); r1 = min(int(rows[-1]) + margin + 1, ny)
        c0 = max(int(cols[0]) - margin, 0); c1 = min(int(cols[-1]) + margin + 1, nx)
        rh, cw = r1 - r0, c1 - c0
        grid_dim = ((rh * cw + threads - 1) // threads, M)
        reg = (np.int32(r0), np.int32(c0), np.int32(rh), np.int32(cw))
        while True:
            while True:
                changed[0] = 0
                for _ in range(6):
                    relax(grid_dim, (threads,), (P, np.int32(ny), np.int32(nx), dx, arr, band_end, changed) + reg)
                if int(changed[0]) == 0:
                    break
            newsp[0] = 0
            finalize(grid_dim, (threads,), (P, np.int32(ny), np.int32(nx), dx, arr, band_end, fin, ros_o, fli_o, dev["spotA"], dev["spotB"], dev["spotign"],
                                            np.float32(spot_rate), np.uint32(seed), np.float32(tmax), spots, nsp, np.int32(0), newsp) + reg)
            if int(newsp[0]) == 0:
                break
    cp.cuda.Stream.null.synchronize()
    return arr.get().reshape(M, ny, nx)
