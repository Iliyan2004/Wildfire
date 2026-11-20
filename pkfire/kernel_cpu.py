"""
Minimum-travel-time fire spread (Finney 2002) with Rothermel rates, a
Richards/Anderson elliptical fire shape per cell, time-varying weather and
stochastic ember spotting.

Each cell, when its arrival time is settled (Dijkstra order), is given a head
rate, head direction and length-to-breadth ratio from the fuel, slope and the
wind of the hour in which it ignited; the fire then travels to its 16
neighbours at the elliptical rate in that edge's direction. Arrival times are
exact for the discretised problem; the 16-neighbour stencil removes most of the
grid-direction bias of an 8-neighbour cellular automaton.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

from . import fuels as fm

# 8 neighbours + 8 knight moves, as (d_col, d_row); rows grow southward
_DC = np.array([1, 1, 0, -1, -1, -1, 0, 1, 2, 2, 1, -1, -2, -2, -1, 1], np.int32)
_DR = np.array([0, 1, 1, 1, 0, -1, -1, -1, 1, -1, 2, 2, -1, 1, -2, -2], np.int32)

M_PER_S_TO_FT_PER_MIN = 196.850394
BTU_FT2_MIN_TO_KW_M2 = 0.18927
MAX_HEAD_ROS = 200.0       # m/min, physical cap


@njit(cache=True, inline="always")
def _cell(i, t, nx, fuel, slope, upaz, cont, hour0, nh, r0t, irt, Bt, Ct, Et, rat, sk, waf, tau,
          wind, windaz, wmod, tdec, cI0, cCBD, emc, wrot):
    """Head ros (m/min), head az (rad), eccentricity, fireline intensity (kW/m) of cell i at time t."""
    row = fuel[i]
    if row == 0:
        return 0.0, 0.0, 0.0, 0.0
    h = hour0 + int(t / 60.0)
    if h >= nh:
        h = nh - 1
    r0 = r0t[h, row] * cont[i]
    if tdec > 0.0:                                   # burn-out / suppression proxy: spread rate fades as the fire ages
        r0 = r0 / (1.0 + t / tdec)
    if r0 <= 1e-4:
        return 0.0, 0.0, 0.0, 0.0
    u = wind[h] * waf[row] * wmod[i] * M_PER_S_TO_FT_PER_MIN
    phiw = Ct[row] * (u ** Bt[row]) * (rat[row] ** (-Et[row])) if u > 0 else 0.0
    tn = slope[i]
    phis = sk[row] * tn * tn
    dw = r0 * phiw
    ds = r0 * phis
    vx = dw * math.sin(windaz[h] + wrot[i]) + ds * math.sin(upaz[i])
    vy = dw * math.cos(windaz[h] + wrot[i]) + ds * math.cos(upaz[i])
    mag = math.sqrt(vx * vx + vy * vy)
    az = math.atan2(vx, vy) if mag > 1e-9 else windaz[h] + wrot[i]
    phie = mag / r0
    ueff = 0.0
    if phie > 0:
        ueff = (phie * rat[row] ** Et[row] / Ct[row]) ** (1.0 / Bt[row])
        cap = 0.9 * irt[h, row]
        if ueff > cap:
            ueff = cap
            phie = Ct[row] * cap ** Bt[row] * rat[row] ** (-Et[row])
    ros = r0 * (1.0 + phie)
    if ros > MAX_HEAD_ROS:
        ros = MAX_HEAD_ROS
    ums = ueff / M_PER_S_TO_FT_PER_MIN        # effective wind in m/s: Anderson's coefficients are per m/s
    lb = 0.936 * math.exp(0.2566 * ums) + 0.461 * math.exp(-0.1548 * ums) - 0.397
    if lb < 1.0:
        lb = 1.0
    if lb > 8.0:
        lb = 8.0
    ecc = math.sqrt(1.0 - 1.0 / (lb * lb))
    inten = irt[h, row] * BTU_FT2_MIN_TO_KW_M2 * ros * tau[row]
    if cCBD[i] > 0.0 and inten >= cI0[i]:                # crown fire: surface fire intense enough to ignite the canopy (Van Wagner 1977)
        ros_c = 11.02 * (wind[h] * 3.6) ** 0.90 * cCBD[i] ** 0.19 * math.exp(-0.17 * emc[h])     # Cruz et al. (2005) active crown rate, m/min
        if ros_c >= 3.0 / cCBD[i]:                         # active crown fire: rate exceeds the critical rate for the canopy
            if ros_c > ros:
                ros = min(ros_c, MAX_HEAD_ROS)
            inten = inten + 150.0 * ros_c                    # canopy fuel consumed (about 0.5 kg/m2 at 18000 kJ/kg) adds to the line intensity
    return ros, az, ecc, inten


@njit(cache=True)
def _up(heap, pos, key, k):
    node = heap[k]
    kv = key[node]
    while k > 0:
        p = (k - 1) >> 1
        if key[heap[p]] <= kv:
            break
        heap[k] = heap[p]
        pos[heap[k]] = k
        k = p
    heap[k] = node
    pos[node] = k


@njit(cache=True)
def _down(heap, pos, key, k, n):
    node = heap[k]
    kv = key[node]
    while True:
        c = 2 * k + 1
        if c >= n:
            break
        if c + 1 < n and key[heap[c + 1]] < key[heap[c]]:
            c += 1
        if key[heap[c]] >= kv:
            break
        heap[k] = heap[c]
        pos[heap[k]] = k
        k = c
    heap[k] = node
    pos[node] = k


@njit(cache=True)
def max_spot_m(hfi, u, A, B):
    """Albini (1983) maximum spotting distance (m) from a wind-driven surface fire: hfi kW/m, u 10 m open wind m/s."""
    if A <= 0.0 or hfi <= 0.0:
        return 0.0
    u = max(u, 0.1)
    E = hfi * A * u ** B
    H = 0.173 * math.sqrt(E)
    if H <= 0.0:
        return 0.0
    h = H ** 0.337 - 1.22
    if not (h > 0.1 and H > h * 1.01):
        return 0.0
    u_h = u * (max(h, 6.0) / 10.0) ** (1.0 / 7.0)
    u_H = u * (H / 10.0) ** (1.0 / 7.0)
    descent = 1.3 * 3.6 * u_h * math.sqrt(h) * (0.362 + math.sqrt(H / h) * 0.5 * math.log(H / h))
    if h < 10.0:
        lofting = 2.78 * u_H * math.sqrt(H)
    else:
        lofting = u_h * math.sqrt(H) * (2.13 + math.log(H / h))
    return min(max(descent + lofting, 0.0), 2000.0)


@njit(cache=True)
def run_mtt(ny, nx, dx, fuel, slope, upaz, cont, hour0, nh, r0t, irt, Bt, Ct, Et, rat, sk, waf, tau,
            spotA, spotB, spot_ign, wind, windaz, ign_idx, ign_t, tmax, spot_rate, seed, dc, dr, max_spots, wmod, tdec, cI0, cCBD, emc, wrot):
    n = ny * nx
    arrival = np.full(n, np.inf, np.float32)
    ros_out = np.zeros(n, np.float32)
    fli_out = np.zeros(n, np.float32)
    done = np.zeros(n, np.uint8)
    heap = np.empty(n, np.int32)
    pos = np.full(n, -1, np.int32)
    hn = 0
    spots = np.full((max_spots, 3), -1.0, np.float64)    # src idx, dst idx, landing time
    nsp = 0
    np.random.seed(seed)

    for k in range(len(ign_idx)):
        i = ign_idx[k]
        if fuel[i] == 0:
            continue
        if ign_t[k] < arrival[i]:
            arrival[i] = ign_t[k]
            if pos[i] < 0:
                heap[hn] = i
                pos[i] = hn
                hn += 1
            _up(heap, pos, arrival, pos[i])

    while hn > 0:
        i = heap[0]
        hn -= 1
        pos[i] = -2
        if hn > 0:
            heap[0] = heap[hn]
            pos[heap[0]] = 0
            _down(heap, pos, arrival, 0, hn)
        t = arrival[i]
        if t > tmax:
            break
        done[i] = 1
        ros, az, ecc, inten = _cell(i, t, nx, fuel, slope, upaz, cont, hour0, nh, r0t, irt, Bt, Ct, Et,
                                    rat, sk, waf, tau, wind, windaz, wmod, tdec, cI0, cCBD, emc, wrot)
        ros_out[i] = ros
        fli_out[i] = inten
        if ros <= 0.0:
            continue
        r = i // nx
        c = i - r * nx
        for m in range(16):
            cc = c + dc[m]
            rr = r + dr[m]
            if cc < 0 or cc >= nx or rr < 0 or rr >= ny:
                continue
            j = rr * nx + cc
            if done[j] == 1 or fuel[j] == 0:
                continue
            stepaz = math.atan2(float(dc[m]), float(-dr[m]))
            cth = math.cos(stepaz - az)
            rate = ros * (1.0 - ecc) / (1.0 - ecc * cth)
            if rate < 1e-3:
                continue
            dist = dx * math.sqrt(float(dc[m] * dc[m] + dr[m] * dr[m]))
            tj = t + dist / rate
            if tj < arrival[j]:
                arrival[j] = tj
                if pos[j] < 0:
                    heap[hn] = j
                    pos[j] = hn
                    hn += 1
                _up(heap, pos, arrival, pos[j])

        # ember spotting (Albini 1983 maximum distance, Schroeder 1969 ignition probability): fuels that shed embers, any intensity
        if spot_rate > 0.0 and spotA[fuel[i]] > 0.0 and inten > 100.0 and nsp < max_spots:
            if np.random.random() < spot_rate * min(inten / 1000.0, 1.0):
                h = hour0 + int(t / 60.0)
                if h >= nh:
                    h = nh - 1
                u10 = wind[h]
                dmax = max_spot_m(inten, u10, spotA[fuel[i]], spotB[fuel[i]])
                if dmax > 30.0:
                    d = min(dmax, dmax * 0.35 * math.exp(0.6 * np.random.randn()))
                    ang = windaz[h] + np.random.randn() * 0.17
                    tc = c + int(round(d * math.sin(ang) / dx))
                    tr = r - int(round(d * math.cos(ang) / dx))
                    if 0 <= tc < nx and 0 <= tr < ny and np.random.random() < spot_ign[h]:
                        j = tr * nx + tc
                        if done[j] == 0 and fuel[j] != 0:
                            tj = t + 3.0 + d / (max(u10, 1.0) * 60.0)
                            if tj < arrival[j] and tj <= tmax:
                                arrival[j] = tj
                                if pos[j] < 0:
                                    heap[hn] = j
                                    pos[j] = hn
                                    hn += 1
                                _up(heap, pos, arrival, pos[j])
                                spots[nsp, 0] = i
                                spots[nsp, 1] = j
                                spots[nsp, 2] = tj
                                nsp += 1
    return arrival, ros_out, fli_out, spots[:nsp]


def simulate(grid, fuel_rows, tables, wind_ms, wind_from_deg, ignitions, hours,
             spot_rate=2e-4, seed=1, hour0=0, max_spots=20000, decay_min=0.0):
    """grid: dict(ny,nx,dx,slope,upaz,cont). ignitions: list of (row, col, t0_min)."""
    ny, nx = grid["ny"], grid["nx"]
    windaz = np.deg2rad((np.asarray(wind_from_deg, np.float64) + 180.0) % 360.0)
    crown = fm.crown_cells(grid, tables, fuel_rows)
    ign_idx = np.array([r * nx + c for r, c, _ in ignitions], np.int64)
    ign_t = np.array([t for _, _, t in ignitions], np.float32)
    arr, ros, fli, spots = run_mtt(
        ny, nx, float(grid["dx"]), fuel_rows.ravel(), grid["slope"].ravel(), grid["upaz"].ravel(),
        grid["cont"].ravel(), hour0, len(wind_ms), tables["r0"], tables["ir"], tables["B"], tables["C"],
        tables["E"], tables["ratio"], tables["slope_k"], tables["waf"], tables["tau"], tables["spot_a"], tables["spot_b"], tables["spot_ign"].astype(np.float64),
        np.asarray(wind_ms, np.float64), windaz, ign_idx, ign_t, float(hours * 60.0), float(spot_rate),
        int(seed), _DC, _DR, int(max_spots), np.ascontiguousarray(grid.get("wmod", np.ones((ny, nx), np.float32)), np.float32).ravel(), float(decay_min), crown[0], crown[1], tables["emc_pct"].astype(np.float64),
        np.ascontiguousarray(grid.get("wrot", np.zeros((ny, nx), np.float32)), np.float32).ravel())
    return arr.reshape(ny, nx), ros.reshape(ny, nx), fli.reshape(ny, nx), spots
