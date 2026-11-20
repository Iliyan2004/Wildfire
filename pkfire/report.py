"""PDF report for a saved run (WeasyPrint): static map, charts, exposure, weather, warnings and provenance."""
from __future__ import annotations

import base64
import html
import json
import os

from . import overlay, run as runmod


def _svg_line(xs, ys, w=330, h=110, color="#e8590c", fill=True, bars=False, xlabel="h", ymax=None):
    if not xs:
        return ""
    ymax = ymax or (max(ys) * 1.1 or 1)
    X = lambda x: 34 + (x / max(xs)) * (w - 40) if max(xs) else 34
    Y = lambda y: 8 + (1 - y / ymax) * (h - 24)
    out = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#f6f8f9;border:1px solid #d9e0e1">']
    for k in range(3):
        v = ymax * k / 2
        out.append(f'<line x1="34" x2="{w-6}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#dde3e5"/><text x="2" y="{Y(v)+3:.1f}" font-size="8" fill="#5d6f74">{v:.0f}</text>')
    if bars:
        bw = max((w - 40) / len(xs) - 1, 1)
        out += [f'<rect x="{X(x)-bw:.1f}" y="{Y(y):.1f}" width="{bw:.1f}" height="{Y(0)-Y(y):.1f}" fill="{color}"/>' for x, y in zip(xs, ys)]
    else:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
        if fill:
            out.append(f'<polygon points="{X(xs[0]):.1f},{Y(0):.1f} {pts} {X(xs[-1]):.1f},{Y(0):.1f}" fill="{color}" opacity=".2"/>')
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
    out.append(f'<text x="{w-30}" y="{h-4}" font-size="8" fill="#5d6f74">{max(xs):.0f} {xlabel}</text></svg>')
    return "".join(out)


def _run_html(rid, full=True):
    d = os.path.join(runmod.RUNS, rid)
    m = json.load(open(os.path.join(d, "meta.json")))
    s = m["stats"]
    png = base64.b64encode(overlay.static_map_png(rid)).decode()
    ar = [[0, 0]] + s["area_series"]
    e = html.escape
    kv = lambda k, v: f"<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>"
    lc = "".join(kv(k, f"{v:,} ha") for k, v in s["by_landcover_ha"].items())
    fu = "".join(kv(k, f"{v:,} ha") for k, v in s["by_fuel_ha"].items())
    ev = "".join(kv(f"+{x['t']/60:.1f} h", x["label"]) for x in m["events"])
    wx = m["weather"]
    ens = ""
    if m.get("ensemble"):
        en = m["ensemble"]
        ens = f"<p><b>Ensemble ({en['members']} members):</b> burned area {en['area_ha_p10']}–{en['area_ha_p90']} ha (10th–90th percentile), median {en['area_ha_median']} ha. Perturbations: wind ±{en['wind_speed_sd_pct']}%, direction ±{en['wind_dir_sd_deg']}°, dead fuel moisture ±{en['dead_moisture_sd_pct']}%, spread-rate error (log-normal, σ {en.get('spread_rate_sd_log', 0)}).</p>"
    expo = ""
    try:
        from . import exposure
        ex = exposure.cached(rid)
        po = ex["population"]
        fr = []
        for k, v in ex["facilities"].items():
            if "error" in v:
                continue
            if k == "power_lines":
                fr.append(kv("Power lines", f"{v['km_burned']} km burned, {v['km_within_1km']} km within 1 km (of {v['km_in_window']} km in window)"))
            else:
                first = next((i for i in v["items"] if i["arrival_min"] is not None), None)
                fr.append(kv(k.replace("_", " ").title(), f"{v['burned']} burned, {v['threatened']} within 1 km (of {v['total_in_window']} in window)" + (f"; first reached +{first['arrival_min']/60:.1f} h ({e(first['name'])})" if first else "")))
        expo = f"<h3>Exposure</h3><table>{kv('People in the burned area', f'{po['in_burn']:,}' if po['available'] else 'n/a')}{kv('People within 1 km of the burn', f'{po['within_1km']:,}' if po['available'] else 'n/a')}{''.join(fr)}</table><p class='note'>{e(ex['note'])}</p>"
    except Exception as exc:
        expo = f"<p class='note'>Exposure not available: {e(str(exc))}</p>"
    warn = "".join(f"<div class='w'>⚠ {e(w)}</div>" for w in m.get("warnings", []))
    return f"""
<h2>Incident {e(rid)}</h2>
<p class="sub">Ignition {m['ignitions'][0][1]:.4f}°N {m['ignitions'][0][0]:.4f}°E · start {e(m['start'])} UTC · {m['hours']} h · {m['cell_m']:.0f} m cells · solver {e(str(s['backend'])).upper()}</p>
<table class="kpi"><tr><td><b>{s['burned_ha']:,}</b><br>ha burned</td><td><b>{s['max_head_ros_m_min']*0.06:.1f}</b><br>km/h peak spread</td><td><b>{s['max_intensity_kw_m']:,.0f}</b><br>kW/m peak intensity</td><td><b>{s['spot_fires']}</b><br>spot fires</td><td><b>{s['builtup_within_1km_ha']:,}</b><br>ha built-up ≤ 1 km</td></tr></table>
<img src="data:image/png;base64,{png}" style="width:100%;max-height:430px;object-fit:contain;background:#0e141a">
<p class="cap">Burn scar coloured by time of arrival (yellow early, dark late). Bounds {m['bounds'][0]:.3f}–{m['bounds'][2]:.3f}°E, {m['bounds'][1]:.3f}–{m['bounds'][3]:.3f}°N.</p>
<div class="row"><div>{_svg_line([p[0]/60 for p in ar],[p[1] for p in ar])}<div class="cap">Fire area (ha) over time</div></div>
<div>{_svg_line([p[0]/60 for p in s['ros_series']],[p[1] for p in s['ros_series']],color='#c92a2a',bars=True)}<div class="cap">Peak spread rate (m/min), per 30 min</div></div></div>
{ens}
<h3>Key events</h3><table>{ev}</table>
<div class="row"><div><h3>Burned by land cover</h3><table>{lc}</table></div><div><h3>Burned by fuel model</h3><table>{fu}</table></div></div>
{expo}
<h3>Weather and fuel moisture</h3><table>{kv('Source', wx['source'])}{kv('Moisture lead-in', wx['conditioning'])}{kv('Peak wind', str(max(wx['wind']))+' m/s')}{kv('Lowest humidity', str(min(wx['rh']))+' %')}{kv('Highest temperature', str(max(wx['temp']))+' °C')}{kv('Driest 1 h fuel moisture', str(min(wx['dead1h']))+' %')}{kv('NDVI used', str(s['ndvi_mean'])+' ('+str(s['ndvi_file'])+')')}</table>
<h3>Warnings and limitations</h3>{warn}"""


_CSS = """@page{size:A4;margin:16mm 14mm;@bottom-center{content:'PAKFIRE incident report • page ' counter(page);font:8pt sans-serif;color:#667}}
body{font:9.6pt/1.45 'DejaVu Sans',sans-serif;color:#1b2a2f}h1{font-size:22pt;margin:0;color:#c4501a}h2{font-size:14pt;color:#c4501a;border-bottom:2px solid #c4501a;padding-bottom:2px}
h3{font-size:10.5pt;margin:12px 0 3px}.sub,.cap{color:#5d6f74;font-size:8.5pt}table{border-collapse:collapse;width:100%;font-size:8.8pt}td{border:1px solid #d9e0e1;padding:3px 6px;vertical-align:top}
.kpi td{text-align:center;font-size:8.5pt;background:#f4f7f7}.kpi b{font-size:13pt}.row{display:flex;gap:14px}.row>div{flex:1}.w{border-left:3px solid #e8892b;background:#fff7ec;padding:4px 9px;margin:4px 0;font-size:8.8pt}
.pb{break-before:page}.note{font-size:8.5pt;color:#5d6f74}"""


def run_pdf(rid):
    import weasyprint
    body = _run_html(rid) + "<p class='note'>Model: Rothermel (1972) surface fire with Albini size classes and Scott &amp; Burgan fuels; minimum-travel-time spread; SRTM 29 m, ESRI 10 m land cover, MODIS NDVI, Open-Meteo weather. Research prototype: not calibrated or validated against observed Pakistani fires. Not for operational decisions.</p>"
    return weasyprint.HTML(string=f"<html><head><meta charset='utf-8'><style>{_CSS}</style></head><body><h1>PAKFIRE</h1><p class='sub'>Wildfire spread incident report</p>{body}</body></html>").write_pdf()


def batch_pdf(ids):
    import weasyprint
    parts = []
    for i, rid in enumerate(ids):
        try:
            parts.append(("<div class='pb'>" if i else "<div>") + _run_html(rid) + "</div>")
        except Exception:
            continue
    return weasyprint.HTML(string=f"<html><head><meta charset='utf-8'><style>{_CSS}</style></head><body><h1>PAKFIRE</h1><p class='sub'>Batch report: {len(parts)} runs</p>{''.join(parts)}</body></html>").write_pdf()
