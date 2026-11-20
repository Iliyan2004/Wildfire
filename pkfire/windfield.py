"""Mass-conserving terrain wind (a diagnostic model of the kind WindNinja uses), written for PAKFIRE and solved on the GPU.

Method (Sherman 1978; Forthofer et al. 2014). An initial wind field is built from the reference 10 m wind with a logarithmic
profile in height above ground. It is then adjusted as little as possible, in the least-squares sense, so that the 3-D flow is
free of divergence around the terrain:

    minimise  integral |u - u0|^2 dV   subject to   div(u) = 0   and no flow through the ground
    =>   u = u0 + grad(phi),      laplace(phi) = - div(u0)

phi is solved on a Cartesian grid whose cells below the terrain are solid (flux through their faces is zero); the lateral and top
boundaries are open (phi = 0 beyond them), so air may enter, leave, go over a ridge or around it. The system is symmetric positive
definite and is solved with Jacobi-preconditioned conjugate gradients; every operation is an array shift, so it runs on the GPU
(CuPy) or, as a fallback, in NumPy.

What is returned is a ratio field: adjusted speed over initial speed, and the change in wind direction, at the lowest model levels
above the ground, for one wind direction (neutral stability, so the ratio does not depend on the wind speed). The fire kernels
multiply the weather wind by the ratio and rotate it by the angle. This is a diagnostic model: no thermal slope winds, no stability,
no flow separation, and staircase terrain at the solver resolution (default 240 m horizontal, 40 m vertical).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

# aerodynamic roughness length z0 (m) by ESRI land-cover class
Z0 = {1: 0.001, 2: 0.9, 4: 0.15, 5: 0.12, 7: 0.9, 8: 0.015, 9: 0.002, 11: 0.06}
DEFAULT_Z0 = 0.1


def z0_from_lc(lc):
    out = np.full(lc.shape, DEFAULT_Z0, np.float32)
    for k, v in Z0.items():
        out[lc == k] = v
    return out


_KERNEL = r'''
extern "C" __global__ void apply_A(const float* __restrict__ p, float* __restrict__ out, const unsigned char* __restrict__ fl, const float* __restrict__ diag,
                                   int nz, int ny, int nx, float wx, float wz) {
    long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)nz * ny * nx;
    if (id >= n) return;
    unsigned char f = fl[id];
    if (!(f & 64)) { out[id] = 0.f; return; }
    int i = id % nx; long t = id / nx; int j = t % ny; int k = t / ny;
    float sx = 0.f, sy = 0.f, sz = 0.f;
    if (f & 1)  sx += p[id + 1];
    if (f & 2)  sx += p[id - 1];
    if (f & 4)  sy += p[id + nx];
    if (f & 8)  sy += p[id - nx];
    if (f & 16) sz += p[id + (long)nx * ny];
    if (f & 32) sz += p[id - (long)nx * ny];
    out[id] = diag[id] * p[id] - wx * (sx + sy) - wz * sz;
}
'''
_kern = {}


def _apply_kernel():
    if "k" not in _kern:
        import cupy as cp
        _kern["k"] = cp.RawKernel(_KERNEL, "apply_A")
    return _kern["k"]


def _xp(backend):
    if backend != "cpu":
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp, "gpu"
        except Exception:
            pass
    return np, "cpu"


def _block(a, k, how="mean"):
    ny, nx = a.shape
    a = a[: ny // k * k, : nx // k * k].reshape(ny // k, k, nx // k, k)
    return a.mean((1, 3)) if how == "mean" else a.max((1, 3))


def solve(dem, dx, from_deg, z0, dxw=240.0, dz=40.0, top_above=700.0, tol=2e-4, maxit=1500, backend="auto", alpha=0.35, level=1):
    """Returns (speed_ratio, rotation_deg, info), both fields on the grid of ``dem`` (rows run north to south).

    dem   elevation (m); dx its cell size (m); from_deg meteorological wind direction (degrees the wind blows FROM);
    z0    roughness length field (m), same shape as dem.
    alpha  Sherman's ratio alpha1/alpha2 of the horizontal to the vertical adjustment weights. Default 0.35: with 1 (neutral) a 400 m hill of
           sigma 2.5 km speeds the wind at its crest by only 4 %, with 0.35 by 20 %, which is what Jackson & Hunt (1975) theory gives (1.22).   level  model level above the ground sampled for the ratio."""
    xp, used = _xp(backend)
    ny0, nx0 = dem.shape
    k = max(int(round(dxw / dx)), 1)
    h = dx * k
    zs = _block(dem.astype(np.float32), k)[::-1].copy()                  # north-up: row index increases northward
    z0c = np.maximum(_block(z0.astype(np.float32), k), 1e-4)[::-1].copy()
    ny, nx = zs.shape
    zmin, zmax = float(zs.min()), float(zs.max())
    nz = int(math.ceil((zmax - zmin + top_above) / dz))
    zc = zmin + (np.arange(nz, dtype=np.float32) + 0.5) * dz
    fluid = zc[:, None, None] > zs[None, :, :]                           # cell centre above the ground
    hgt = np.maximum(zc[:, None, None] - zs[None, :, :], 0.0)
    to = math.radians((from_deg + 180.0) % 360.0)
    ex, ey = math.sin(to), math.cos(to)                                  # unit vector the wind blows towards (east, north)
    prof = np.log(np.maximum(np.minimum(hgt, 400.0), 1.05 * z0c[None]) / z0c[None]) / np.log(10.0 / z0c[None])
    prof = np.where(fluid, prof, 0.0).astype(np.float32)

    F = xp.asarray(fluid)
    u0 = xp.asarray(prof * np.float32(ex))
    v0 = xp.asarray(prof * np.float32(ey))
    f32 = xp.float32
    # open faces between fluid cells (internal) and open boundary faces (lateral and top), as 0/1 floats
    def shift_and(a, axis, direction):
        """a & neighbour in the given direction (+1: the next index), False past the array edge."""
        out = xp.zeros_like(a)
        sl_a = [slice(None)] * 3; sl_n = [slice(None)] * 3
        if direction > 0:
            sl_a[axis] = slice(0, -1); sl_n[axis] = slice(1, None)
        else:
            sl_a[axis] = slice(1, None); sl_n[axis] = slice(0, -1)
        out[tuple(sl_a)] = a[tuple(sl_a)] & a[tuple(sl_n)]
        return out
    oxp, oxm = shift_and(F, 2, +1), shift_and(F, 2, -1)
    oyp, oym = shift_and(F, 1, +1), shift_and(F, 1, -1)
    ozp, ozm = shift_and(F, 0, +1), shift_and(F, 0, -1)
    # Dirichlet (open) boundary faces count in the diagonal but have no neighbour term
    bxp = xp.zeros_like(F); bxp[:, :, -1] = F[:, :, -1]
    bxm = xp.zeros_like(F); bxm[:, :, 0] = F[:, :, 0]
    byp = xp.zeros_like(F); byp[:, -1, :] = F[:, -1, :]
    bym = xp.zeros_like(F); bym[:, 0, :] = F[:, 0, :]
    bzp = xp.zeros_like(F); bzp[-1] = F[-1]
    wx, wz = 1.0 / (h * h), alpha * alpha / (dz * dz)         # minimising  |du_h|^2 + |dw|^2 / alpha^2  gives  laplace_h(phi) + alpha^2 phi_zz = -div(u0)
    cx = (oxp | bxp).astype(f32) + (oxm | bxm).astype(f32)
    cy = (oyp | byp).astype(f32) + (oym | bym).astype(f32)
    cz = (ozp | bzp).astype(f32) + ozm.astype(f32)
    diag = (wx * (cx + cy) + wz * cz).astype(f32)
    active = F & (diag > 0)
    inv_diag = xp.where(active, 1.0 / xp.maximum(diag, 1e-12), 0.0).astype(f32)
    oxp_f, oxm_f, oyp_f, oym_f, ozp_f, ozm_f = (a.astype(f32) for a in (oxp, oxm, oyp, oym, ozp, ozm))

    def face_u(a, axis):                                                 # face-centred initial component between cell and its + neighbour
        out = xp.zeros_like(a)
        sl_a = [slice(None)] * 3; sl_n = [slice(None)] * 3
        sl_a[axis] = slice(0, -1); sl_n[axis] = slice(1, None)
        out[tuple(sl_a)] = 0.5 * (a[tuple(sl_a)] + a[tuple(sl_n)])
        return out
    fxp = face_u(u0, 2) * oxp_f                                          # east face velocity (internal fluid-fluid faces only)
    fxp[:, :, -1] = u0[:, :, -1] * bxp[:, :, -1]                         # open boundary: the initial wind crosses it
    fxm = xp.zeros_like(u0); fxm[:, :, 1:] = fxp[:, :, :-1]; fxm[:, :, 0] = u0[:, :, 0] * bxm[:, :, 0]
    fyp = face_u(v0, 1) * oyp_f; fyp[:, -1, :] = v0[:, -1, :] * byp[:, -1, :]
    fym = xp.zeros_like(v0); fym[:, 1:, :] = fyp[:, :-1, :]; fym[:, 0, :] = v0[:, 0, :] * bym[:, 0, :]
    fzp = xp.zeros_like(u0)                                              # w0 = 0 everywhere
    div0 = ((fxp - fxm) / h + (fyp - fym) / h + (fzp - fzp) / dz) * active.astype(f32)

    flags = (oxp_f.astype(xp.uint8) | (oxm_f.astype(xp.uint8) << 1) | (oyp_f.astype(xp.uint8) << 2) | (oym_f.astype(xp.uint8) << 3)
             | (ozp_f.astype(xp.uint8) << 4) | (ozm_f.astype(xp.uint8) << 5) | (active.astype(xp.uint8) << 6)).astype(xp.uint8)
    use_kernel = used == "gpu"
    if use_kernel:
        kern = _apply_kernel()
        npts = nz * ny * nx
        grid_dim = ((npts + 255) // 256,)
        wxf, wzf = np.float32(wx), np.float32(wz)

    def A(p):
        if use_kernel:
            out = xp.empty_like(p)
            kern(grid_dim, (256,), (p, out, flags, diag, np.int32(nz), np.int32(ny), np.int32(nx), wxf, wzf))
            return out
        s = xp.zeros_like(p)
        s[:, :, :-1] += oxp_f[:, :, :-1] * p[:, :, 1:] * wx
        s[:, :, 1:] += oxm_f[:, :, 1:] * p[:, :, :-1] * wx
        s[:, :-1, :] += oyp_f[:, :-1, :] * p[:, 1:, :] * wx
        s[:, 1:, :] += oym_f[:, 1:, :] * p[:, :-1, :] * wx
        s[:-1] += ozp_f[:-1] * p[1:] * wz
        s[1:] += ozm_f[1:] * p[:-1] * wz
        return (diag * p - s) * active

    # conjugate gradients on  A phi = div0   (A = -laplace, so  laplace phi = -div0)
    b = div0
    x = xp.zeros_like(b)
    r = b.copy()
    z = inv_diag * r
    p = z.copy()
    if use_kernel:
        import cupy as cp
        upd = cp.ElementwiseKernel("float32 s, float32 inv, float32 pp, float32 Ap", "float32 x, float32 r, float32 z",
                                   "x += s * pp; r -= s * Ap; z = inv * r;", "cg_update")
        pnew = cp.ElementwiseKernel("float32 beta, float32 z", "float32 p", "p = z + beta * p;", "cg_pupdate")
        dot = lambda u, v: float(cp.vdot(u.ravel(), v.ravel()))
    else:
        dot = lambda u, v: float((u * v).sum())
    rz = dot(r, z)
    bn = math.sqrt(dot(b, b)) + 1e-30
    it = 0
    res = 1.0
    for it in range(1, maxit + 1):
        Ap = A(p)
        step = rz / (dot(p, Ap) + 1e-30)
        if use_kernel:
            upd(np.float32(step), inv_diag, p, Ap, x, r, z)
        else:
            x += step * p
            r -= step * Ap
            z = inv_diag * r
        res = math.sqrt(dot(r, r)) / bn
        if res < tol:
            break
        rz2 = dot(r, z)
        if use_kernel:
            pnew(np.float32(rz2 / rz), z, p)
        else:
            p = z + (rz2 / rz) * p
        rz = rz2
    # adjusted face velocities and the divergence left over
    pxp = xp.zeros_like(x); pxp[:, :, :-1] = x[:, :, 1:]
    pyp = xp.zeros_like(x); pyp[:, :-1, :] = x[:, 1:, :]
    pzp = xp.zeros_like(x); pzp[:-1] = x[1:]
    uxp = fxp + oxp_f * (pxp - x) / h - bxp.astype(f32) * x / h
    uyp = fyp + oyp_f * (pyp - x) / h - byp.astype(f32) * x / h
    uxm = xp.zeros_like(x); uxm[:, :, 1:] = uxp[:, :, :-1]; uxm[:, :, 0] = fxm[:, :, 0] + bxm[:, :, 0].astype(f32) * x[:, :, 0] / h
    uym = xp.zeros_like(x); uym[:, 1:, :] = uyp[:, :-1, :]; uym[:, 0, :] = fym[:, 0, :] + bym[:, 0, :].astype(f32) * x[:, 0, :] / h
    a2 = alpha * alpha
    uzp = a2 * (ozp_f * (pzp - x) / dz - bzp.astype(f32) * x / dz)
    uzm = xp.zeros_like(x); uzm[1:] = uzp[:-1]
    div1 = ((uxp - uxm) / h + (uyp - uym) / h + (uzp - uzm) / dz) * active.astype(f32)
    ucell = 0.5 * (uxp + uxm)
    vcell = 0.5 * (uyp + uym)
    div0n, div1n = float(xp.abs(div0).max()), float(xp.abs(div1).max())
    ucell, vcell, u0c, v0c, Fn = (a.get() if xp is not np else a for a in (ucell, vcell, u0, v0, F))

    # sample at the second fluid level above the ground of each column (30-80 m above ground with 40 m layers)
    first = np.argmax(Fn, axis=0)
    kk = np.minimum(first + int(level), nz - 1)
    jj, ii = np.indices(first.shape)
    ua, va, ub, vb = ucell[kk, jj, ii], vcell[kk, jj, ii], u0c[kk, jj, ii], v0c[kk, jj, ii]
    sp_a, sp_b = np.hypot(ua, va), np.hypot(ub, vb)
    ratio = np.where(sp_b > 1e-6, sp_a / np.maximum(sp_b, 1e-6), 1.0)
    rot = np.degrees(np.arctan2(ub * va - vb * ua, ub * ua + vb * va))        # anticlockwise from the initial direction (north-up axes)
    ratio = np.clip(ratio, 0.3, 3.0).astype(np.float32)
    rot = np.clip(rot, -75.0, 75.0).astype(np.float32)
    ratio = ndimage.uniform_filter(ratio, 3, mode="nearest")[::-1]
    rot = ndimage.uniform_filter(rot, 3, mode="nearest")[::-1]               # back to north-up rows = image rows
    full_r = ndimage.zoom(ratio, (ny0 / ratio.shape[0], nx0 / ratio.shape[1]), order=1)
    full_a = ndimage.zoom(rot, (ny0 / rot.shape[0], nx0 / rot.shape[1]), order=1)
    info = dict(backend=used, cells=int(nz * ny * nx), grid=[nz, ny, nx], cell_m=[h, dz], iterations=it, residual=res,
                max_div_before=div0n, max_div_after=div1n)
    return full_r.astype(np.float32), full_a.astype(np.float32), info

