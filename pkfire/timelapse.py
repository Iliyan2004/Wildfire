"""Server-side time-lapse video (MP4, H.264) of a saved run: hillshade background, the burn scar by age, a glowing front, spot fires,
the spread axis, a clock, burned area, wind arrow and a scale bar. Frames are drawn with NumPy/Pillow and encoded with the ffmpeg
that ships with imageio-ffmpeg. No map tiles: the background is the model's own terrain, so the video needs no internet."""
from __future__ import annotations

import datetime as dt
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from . import overlay, run as runmod

PKT = dt.timedelta(hours=5)


def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _lut():
    """Colour by age in minutes (0.5 min steps, 0-240) -> RGBA, as in the 3D scene."""
    n = 481
    lut = np.zeros((n, 4), np.float32)
    lerp = lambda a, b, t: a + (b - a) * min(max(t, 0.0), 1.0)
    for k in range(n):
        a = k / 2
        if a < 4:   t = a / 4;           c = (255, lerp(246, 160, t), lerp(190, 40, t), 1.0)
        elif a < 15: t = (a - 4) / 11;   c = (lerp(255, 232, t), lerp(160, 76, t), lerp(40, 18, t), 0.97)
        elif a < 60: t = (a - 15) / 45;  c = (lerp(232, 160, t), lerp(76, 24, t), lerp(18, 14, t), lerp(0.95, 0.82, t))
        else:        t = (a - 60) / 180; c = (lerp(160, 44, t), lerp(24, 28, t), lerp(14, 22, t), lerp(0.82, 0.70, t))
        lut[k] = c
    return lut


def render(run_id, seconds=24.0, fps=24, width=1280, out_path=None, progress=lambda *_: None):
    d = os.path.join(runmod.RUNS, run_id)
    meta, arr, inten, _ = overlay.load(run_id)
    w, h = meta["size"]
    never, ts = meta["never"], meta["t_scale"]
    hs_path = os.path.join(d, "hillshade.bin")
    hs = np.fromfile(hs_path, np.uint8).reshape(h, w).astype(np.float32) / 255.0 if os.path.exists(hs_path) else np.full((h, w), 0.5, np.float32)
    burnt = arr < never
    ys, xs = np.nonzero(burnt)
    pad = int(0.12 * max(xs.max() - xs.min(), ys.max() - ys.min(), 60)) + 8
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)
    # keep the whole fire in frame; the camera is fixed (the follow camera lives in the 3D scene)
    cw, ch = x1 - x0, y1 - y0
    scale = width / cw
    H = int(round(ch * scale / 2) * 2); W = int(round(width / 2) * 2)
    A = arr[y0:y1, x0:x1].astype(np.float32) / ts            # minutes
    A[~burnt[y0:y1, x0:x1]] = np.inf
    I = inten[y0:y1, x0:x1].astype(np.float32)
    base = hs[y0:y1, x0:x1]
    base_rgb = np.stack([0.10 + 0.34 * base, 0.12 + 0.34 * base, 0.14 + 0.34 * base], -1)         # dark, slightly cool terrain
    lut = _lut()
    t_end = float(meta["t_end_min"])
    n_frames = int(seconds * fps)
    times = np.linspace(0, t_end, n_frames)
    wx = meta["weather"]
    start = dt.datetime.fromisoformat(meta["start"])
    f_big, f_mid, f_sm = _font(30), _font(18), _font(14)
    # spread axis and spots in cropped pixel coordinates
    Wb, Sb, Eb, Nb = meta["bounds"]
    def px(lon, lat):
        return ((lon - Wb) / (Eb - Wb) * w - x0) * scale, ((Nb - lat) / (Nb - Sb) * h - y0) * scale
    axis = [(t, *px(lo, la)) for t, lo, la, *_ in meta["axis"]]
    spots = [(t, *px(lo, la)) for lo, la, t in meta["spots"]]
    ign = [px(*c) for c in meta["ignitions"]]
    import imageio_ffmpeg, subprocess
    out_path = out_path or os.path.join(d, f"timelapse_{int(seconds)}s.mp4")
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                           "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
                          stdin=subprocess.PIPE)
    ha_per_px = meta["stats"]["burned_ha"] / max(int(burnt.sum()), 1)
    for fi, T in enumerate(times):
        age = T - A
        on = age >= 0
        idx = np.clip(np.where(on, age, 0) * 2, 0, 480).astype(np.int32)
        c = lut[idx]
        q = np.clip(I / 255.0 * 3.0, 0.0, 1.0)[..., None]
        rgb = c[..., :3] / 255.0
        front = on & (age < 15)
        rgb = np.where(front[..., None], np.clip(rgb * (0.72 + 0.28 * q), 0, 1), rgb)
        al = np.where(on, c[..., 3], 0.0)[..., None]
        img = base_rgb * (1 - al) + rgb * al
        glow = ndimage.gaussian_filter((front & (age < 8)).astype(np.float32) * (0.4 + 0.6 * q[..., 0]), 3.0)
        img = img + glow[..., None] * np.array([1.0, 0.55, 0.15], np.float32) * 1.4
        frame = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        dr = ImageDraw.Draw(frame, "RGBA")
        pts = [(x, y) for t, x, y in axis if t <= T]
        if len(pts) > 1:
            dr.line(pts, fill=(255, 226, 122, 235), width=3)
        for t, x, y in spots:
            if t <= T:
                dr.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(76, 201, 255, 255), width=2)
        for x, y in ign:
            dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 59, 26, 255), outline=(255, 255, 255, 255))
        clock = (start + dt.timedelta(minutes=float(T)) + PKT).strftime("%Y-%m-%d  %H:%M PKT")
        hh = min(int(T // 60), len(wx["wind"]) - 1)
        area = float((A <= T).sum()) * ha_per_px
        dr.rectangle([0, 0, W, 74], fill=(10, 14, 18, 190))
        dr.text((18, 10), clock, fill=(255, 255, 255, 255), font=f_big)
        dr.text((18, 46), f"+{int(T // 60)}h {int(T % 60):02d}m   |   {area:,.0f} ha burned   |   wind {wx['wind'][hh]} m/s   {wx['temp'][hh]:.0f} °C   RH {wx['rh'][hh]:.0f}%", fill=(210, 220, 230, 255), font=f_sm)
        ax, ay = W - 60, 38
        to = math.radians(wx["wdir"][hh] + 180)
        dr.ellipse([ax - 24, ay - 24, ax + 24, ay + 24], outline=(120, 140, 160, 255), width=2)
        dr.line([ax - 16 * math.sin(to), ay + 16 * math.cos(to), ax + 16 * math.sin(to), ay - 16 * math.cos(to)], fill=(76, 201, 255, 255), width=3)
        dr.polygon([(ax + 18 * math.sin(to), ay - 18 * math.cos(to)), (ax + 7 * math.sin(to + 0.7), ay - 7 * math.cos(to + 0.7)), (ax + 7 * math.sin(to - 0.7), ay - 7 * math.cos(to - 0.7))], fill=(76, 201, 255, 255))
        # scale bar (about 5 km or 1 km)
        px_m = (Eb - Wb) * 111320.0 * math.cos(math.radians((Nb + Sb) / 2)) / w            # metres per export pixel
        px_per_km = scale * 1000.0 / px_m
        km = 5 if px_per_km * 5 < W * 0.4 else 1
        dr.line([(18, H - 24), (18 + px_per_km * km, H - 24)], fill=(255, 255, 255, 255), width=3)
        dr.text((18, H - 48), f"{km} km", fill=(255, 255, 255, 255), font=f_sm)
        dr.text((W - 330, H - 26), "PAKFIRE scenario, not a forecast", fill=(200, 210, 220, 200), font=f_sm)
        ff.stdin.write(np.asarray(frame, np.uint8).tobytes())
        if fi % 24 == 0:
            progress(fi / n_frames)
    ff.stdin.close()
    ff.wait()
    if ff.returncode:
        raise RuntimeError("ffmpeg failed")
    return out_path
