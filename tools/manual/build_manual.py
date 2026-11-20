"""Builds docs/PAKFIRE_User_Manual.html and .pdf (WeasyPrint) from the screenshots in docs/manual/shots (tools/manual/capture.py).
   ~/flood/venv/bin/python tools/manual/build_manual.py"""
import os, re, html
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SHOTS = os.path.join(ROOT, "docs", "manual", "shots")
TOC = []


def chap(n, title, key):
    TOC.append((1, f"{n} {title}", key)); return f'<h1 class="chap" id="{key}"><span class="num">{n}</span>{title}</h1>'


def sec(n, title, key):
    TOC.append((2, f"{n} {title}", key)); return f'<h2 id="{key}">{n} {title}</h2>'


def steps(*a): return '<ol class="steps">' + "".join(f"<li>{x}</li>" for x in a) + "</ol>"
def tip(t, kind="", head="Tip"): return f'<div class="callout {kind}"><b class="t">{head}</b>{t}</div>'
def warn(t): return tip(t, "warn", "Important")
def note(t): return tip(t, "note", "Note")


def shot(name, caption, legend=()):
    p = os.path.join(SHOTS, name + ".png")
    if not os.path.exists(p):
        return f'<p class="small"><i>[screenshot {name} missing]</i></p>'
    lg = ('<ol class="legend">' + "".join(f"<li>{x}</li>" for x in legend) + "</ol>") if legend else ""
    return f'<figure><img class="shot" src="file://{p}"><figcaption>{caption}{lg}</figcaption></figure>'


def table(head, rows):
    return "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


FLOW = '''<svg viewBox="0 0 640 210" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Nimbus Sans, sans-serif">
<rect x="230" y="8" width="180" height="34" rx="5" fill="#2a1610"/><text x="320" y="30" fill="#fff" font-size="12.5" text-anchor="middle" font-weight="700">What do you want to know?</text>
<g stroke="#a33c14" stroke-width="1.4" fill="none"><path d="M320 42 V60 H105 V80"/><path d="M320 60 H320 V80"/><path d="M320 60 H535 V80"/></g>
<g font-size="10.5" fill="#2a1610" text-anchor="middle">
<rect x="25" y="80" width="160" height="50" rx="4" fill="#fff" stroke="#d4440f"/><text x="105" y="100">Roughly how far a fire</text><text x="105" y="114">could run, right now</text>
<rect x="240" y="80" width="160" height="50" rx="4" fill="#fff" stroke="#d4440f"/><text x="320" y="100">When it arrives, how hot,</text><text x="320" y="114">who is in the way, what helps</text>
<rect x="455" y="80" width="160" height="50" rx="4" fill="#fff" stroke="#d4440f"/><text x="535" y="100">Where fires are burning</text><text x="535" y="114">in Pakistan today</text></g>
<g stroke="#a33c14" stroke-width="1.4"><path d="M105 130 V150"/><path d="M320 130 V150"/><path d="M535 130 V150"/></g>
<g font-size="11.5" fill="#fff" text-anchor="middle" font-weight="700">
<rect x="25" y="150" width="160" height="46" rx="4" fill="#ff6a2b"/><text x="105" y="170">PAKFIRE Instant</text><text x="105" y="186" font-weight="400" font-size="10">about a second</text>
<rect x="240" y="150" width="160" height="46" rx="4" fill="#a33c14"/><text x="320" y="170">PAKFIRE Physics</text><text x="320" y="186" font-weight="400" font-size="10">seconds to a few minutes</text>
<rect x="455" y="150" width="160" height="46" rx="4" fill="#2a1610"/><text x="535" y="170">Discover tab</text><text x="535" y="186" font-weight="400" font-size="10">live satellite data</text></g></svg>'''

ARRIVAL = '''<svg viewBox="0 0 640 150" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Nimbus Sans, sans-serif" font-size="10.5" fill="#2a1610">
<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#fff3b0"/><stop offset=".35" stop-color="#ffb43b"/><stop offset=".7" stop-color="#ff6a2b"/><stop offset="1" stop-color="#7a1d0c"/></linearGradient></defs>
<rect x="40" y="30" width="560" height="22" fill="url(#g)" rx="3"/><text x="40" y="22">Time of arrival</text>
<text x="40" y="70">ignition</text><text x="600" y="70" text-anchor="end">end of run</text>
<text x="40" y="104">Intensity (kW/m) and what it means for crews (the usual fire-behaviour interpretation chart):</text>
<g font-size="10"><text x="40" y="124">&lt; 350: hand tools at the head</text><text x="210" y="124">350-1,700: equipment, not hand tools</text><text x="420" y="124">&gt; 1,700: head attack fails; &gt; 3,500 crowning, spotting</text></g></svg>'''

body = []
A = body.append
A(f'''<section class="cover"><div class="brand">PAKFIRE</div><h1>User Manual</h1>
<div class="sub">From an ignition point to who is in the fire's path. A step-by-step guide to simulating wildfire spread anywhere in Pakistan, reading the result, and testing fuel breaks.</div>
<svg class="wave" viewBox="0 0 300 140"><path d="M0 140 C40 100 60 120 90 70 C110 40 120 80 140 30 C150 60 175 50 185 10 C200 60 230 70 250 40 C260 80 290 90 300 60 L300 140 Z" fill="#ff6a2b" opacity=".55"/><path d="M0 140 C50 120 80 130 110 95 C130 75 150 110 170 70 C190 100 220 100 245 80 C265 110 290 110 300 100 L300 140 Z" fill="#ffb43b" opacity=".6"/></svg>
<div class="meta">PAKFIRE &middot; November 2026<br>Pakistan Wildfire Spread Simulation, Exposure &amp; Response Analytics<br>AI Geo Navigators, Islamabad, Pakistan</div></section>''')
A("<!--TOC-->")

A(chap(1, "Welcome to PAKFIRE", "c1"))
A('<p class="lead">PAKFIRE answers four questions about a wildfire in one place: where it will go and when, how hot it will burn, who and what is in its way, and whether a fuel break or treatment would help. You place an ignition on a map of Pakistan, choose the weather, and PAKFIRE does the rest.</p>')
A(sec("1.1", "What PAKFIRE does", "s11"))
A("<ul><li><b>Simulates fire spread.</b> A physical model (Rothermel surface fire, minimum travel time spread) runs on the GPU over the real terrain, fuel and hourly weather of the place, with ember spotting, crown fire and wind shaped by the terrain.</li><li><b>Counts who is exposed.</b> People (WorldPop scaled to the 2023 census), houses by type, informal settlements, hospitals, schools and power lines in and near the burned area, with the time the fire reaches them.</li><li><b>Plans evacuation.</b> Routes out of the fire's path and from the nearest fire station, over the road network, with the time each route is cut.</li><li><b>Tests mitigation.</b> Draw a fuel break or a treated area and compare the fire with and without it.</li><li><b>Works from real data.</b> Elevation, 10 m land cover, vegetation greenness, satellite fire detections and weather are loaded automatically.</li></ul>")
A(sec("1.2", "The modes", "s12"))
A('<div class="modecard"><h3><span class="pill org">Instant</span> A quick look</h3>Click a place, pick a weather scenario, and see in about a second how far a fire could run in 1, 3 and 6 hours. Good for comparing places and weather quickly.</div>')
A('<div class="modecard"><h3><span class="pill">Physics</span> The full fire model</h3>Runs the full model for up to 120 hours of real or typed weather. Gives the time of arrival and intensity everywhere, an animated 3D fire, exposure and evacuation, ensembles of many runs, and mitigation comparisons. A run takes seconds to a few minutes.</div>')
A('<div class="modecard"><h3><span class="pill navy">Discover</span> Fires burning now</h3>A tab inside Physics: districts ranked by NASA satellite fire detections, the strongest recent hotspots, and live monitoring of a place with automatic scenarios.</div>')
A(sec("1.3", "Which mode should I use?", "s13"))
A(f'<figure>{FLOW}<figcaption>Choosing a mode. When in doubt, start with Instant to find the places that matter, then run Physics on them.</figcaption></figure>')
A(sec("1.4", "What PAKFIRE can and cannot tell you", "s14"))
A("<p>PAKFIRE is a decision-support and research tool. It is good at comparing scenarios, ranking places and weather, and showing how a fire could develop. Please keep these limits in mind:</p><ul><li><b>Results are scenarios, not forecasts.</b> A run shows what happens for the weather and settings you chose.</li><li><b>Burned area is about right, shape less so.</b> Calibrated on satellite-observed fires in Pakistan (2012-2026), the model gets the burned area to within about a factor of two on held-back fires, but the shape and direction of the burn are only slightly better than an equal-area circle.</li><li><b>Fuel is mapped at 10 m and simulated at 50-100 m.</b> Small firebreaks such as paths and narrow fields are not seen.</li><li><b>Crown fire uses assumed canopy values.</b> Pine forest can crown; the canopy figures are estimates.</li><li><b>Weather is for one point.</b> Wind is shaped by the terrain, but there is no weather model behind it.</li><li><b>PAKFIRE does not replace official warnings.</b></li></ul>")
A(sec("1.5", "Before you start", "s15"))
A("<ul><li>A modern web browser (Chrome, Edge, Firefox or Safari) with graphics acceleration switched on; the 3D views use WebGL.</li><li>An ignition inside <b>Pakistan</b>: the data covers latitude 23.5-37.2 and longitude 60.8-77.9.</li><li>For live weather and satellite detections the server needs an internet connection; without it you can type the weather yourself.</li></ul>")

A(chap(2, "A tour of the screen", "c2"))
A("<p>The Physics page uses one layout: a map that fills the screen, a toolbar at the top, a panel with tabs on the right and the 3D fire view in a card at the bottom left. Learn it once.</p>")
A(sec("2.1", "The welcome card", "s21"))
A(shot("welcome", "<b>The welcome card</b> opens the first time you visit in a browser session.", ["<b>Search a place</b> puts the cursor in the search box.", "<b>Place an ignition</b> starts the point tool: click the map.", "<b>Start from a satellite hotspot</b> lists the strongest detections."]))
A(sec("2.2", "The main screen", "s22"))
A(shot("screen", "<b>The main screen.</b>", ["Search box: a district or a place in Pakistan.", "Basemap: Dark (default), Satellite or Streets.", "Modes: Instant, Physics and the user manual.", "Queue: runs waiting or running; cancel from here.", "Light or dark theme.", "Drawing tools: area, line, point, delete.", "Map tools: satellite fire detections, districts shaded by detections, clear overlay, export PNG.", "Tabs of the right panel: Fire, Runs, Impact, Mitigation, Discover.", "Brings back the 3D fire view when hidden."]))
A(table(["Part of the screen", "What it does"], [["Top toolbar", "Search; basemap; mode buttons; the job queue; light or dark theme."], ["Map", "Pakistan only: outside the country is darkened and the map cannot be moved away."], ["Right panel", "Everything you set and read, in five tabs. The tab on its left edge hides or shows it."], ["3D fire view", "The animated fire of the open run. <b>Expand</b> fills the screen, <b>Hide</b> puts it away, <b>Full Page</b> opens it in a new tab."]]))
A(sec("2.3", "Light theme", "s23"))
A(shot("light", "The light theme. The choice is remembered in your browser.", ["Switches between dark and light."]))

A(chap(3, "Placing the fire", "c3"))
A("<p>Every run starts with an ignition point. An area is optional: it sets the size of the simulation window.</p>")
A(sec("3.1", "Place an ignition", "s31"))
A(steps("Open the <b>Fire</b> tab and click <b>Place ignition</b>.", "Click the map where the fire starts. Drag the point to move it.", "To start several fires at once, place more points: the extra ignitions are listed."))
A(shot("fire_tab", "The Fire tab with an ignition in Margalla Hills, Islamabad.", ["<b>Place ignition</b>.", "<b>Draw area</b>: an optional polygon that sets the simulation window.", "The ignition, its district and the window.", "Weather and fuel context: fire-weather danger, current weather, rain, dead fuel moisture, land cover and elevation around the point."]))
A(sec("3.2", "Search, upload or start from a detection", "s32"))
A(table(["You do", "PAKFIRE does"], [["Search a district", "Uses the district as the area and places an ignition at its centre."], ["Search a place", "Flies the map there; then place the ignition."], ["Upload a file", "A GeoJSON or zipped shapefile with a point (the ignition) or a polygon (the area)."], ["Pick a satellite hotspot (Discover tab)", "Starts the fire at a NASA VIIRS detection of the last days."], ["Share link", "<b>Link</b> copies an address that restores the ignition and the weather settings."]]))
A(sec("3.3", "Area and window limits", "s33"))
A("<p>The simulation window is a square around the ignition: from your drawn area, or &plusmn;25, 40, 60 or 100 km (cells of 50 to 100 m). The largest window is 200 km across; a larger drawn area is cut to it.</p>")
A(warn("The ignition must be in burnable fuel. On bare ground, water or a city centre PAKFIRE moves it to the nearest fuel within 600 m, or tells you there is none."))

A(chap(4, "PAKFIRE Instant", "c4"))
A("<p>Instant shows, in about a second, how far a fire could run in 1, 3 and 6 hours from a point you click, for a weather scenario you choose. It is the same physical engine on a small 24 km window with the weather held constant.</p>")
A(sec("4.1", "Step by step", "s41"))
A(steps("Open <b>Instant</b> from the start page or the top toolbar (address <code>/instant</code>).", "In the welcome card, type a place, choose a scenario and click <b>Show fire spread</b>; or close the card and click the map.", "Change the wind, its direction, the humidity or the duration: the result updates by itself."))
A(shot("instant_welcome", "The Instant welcome card.", ["Your location.", "Weather scenario.", "Start."]))
A(sec("4.2", "Reading the result", "s42"))
A(shot("instant_result", "An Instant result for Margalla Hills, Islamabad.", ["Search another place.", "Scenarios: Calm, Breezy, Strong wind, Extreme dry, or the live forecast.", "Wind speed.", "Wind direction (the arrow shows where it blows to).", "Humidity.", "Duration.", "Burned area after each hour mark, spread rate at the ignition, weather used.", "Legend of the fire fronts.", "Opens the same place in the full model."]))
A(table(["Scenario", "Wind", "Humidity", "Air"], [["Calm", "2 m/s", "45 %", "28 &deg;C"], ["Breezy", "6 m/s", "30 %", "32 &deg;C"], ["Strong wind", "12 m/s", "18 %", "36 &deg;C"], ["Extreme, dry", "18 m/s", "10 %", "40 &deg;C"], ["Live weather", "forecast for the place and hour", "", ""]]))
A(sec("4.3", "How far to trust it", "s43"))
A(warn("Instant holds the weather constant, has no spotting and a small window. Use it to compare places and weather and to choose where to look; for any decision run PAKFIRE Physics."))

A(chap(5, "PAKFIRE Physics", "c5"))
A("<p>Physics runs the full model. You place the fire, describe the weather and the window, and PAKFIRE computes when the fire reaches every cell and how hot it burns there, then what that means for people and roads.</p>")
A(sec("5.1", "The Fire tab: weather and timing", "s51"))
A(shot("fire_weather", "Weather and timing.", ["Source: live, forecast or archived hourly weather (Open-Meteo, with MET Norway as fallback) with a 48 h lead-in for fuel moisture, or weather you type.", "Start date and time (Pakistan time).", "Duration, 1 to 120 hours.", "Preview the weather and the fuel moisture it implies."]))
A("<p>With <b>Set the weather yourself</b> you type the wind, its direction (or click the compass), temperature and humidity; they are held for the whole run.</p>")
A(sec("5.2", "Window, rates and options", "s52"))
A(shot("fire_solver", "Window and solver.", ["Window: from the drawn area or a fixed size.", "Rates: calibrated to satellite-observed fires (default) or uncalibrated physics.", "Ember spotting in forest fuels.", "Start from satellite detections: the run begins at the first detection near the ignition and treats every detection within 6 km as burning from the time it was seen.", "<b>Run fire model</b>."]))
A(table(["Setting", "What it does"], [["Backend (Advanced)", "GPU first, GPU only or CPU. The result is the same."], ["Random seed (Advanced)", "Changes where embers land. <b>Run again with a new seed</b> after a run does the same."], ["Cell size (Advanced)", "0 chooses it from the window size."]]))
A(sec("5.3", "Running the model", "s53"))
A(steps("Check the ignition, weather and window in the Fire tab.", "Click <b>Run fire model</b>. A progress bar shows the stage: scene, weather, fuel, spread, outputs.", "When it finishes the 3D fire view opens and the numbers appear under the button."))
A(note("Runs wait their turn: the <b>Queue</b> label in the toolbar shows how many are waiting or running, and can cancel them. Cancelling acts between stages."))
A(sec("5.4", "Reading the result", "s54"))
A(shot("result", "A finished run.", ["Burned area, spread rate, intensity, spot fires, crown fire and warnings.", "<b>Expand</b> the 3D view.", "<b>Hide</b> it.", "Open it as a full page.", "Legend of the map layer."]))
A(sec("5.5", "The 3D fire view", "s55"))
A(shot("view3d", "The 3D view, expanded.", ["Display: fire age, arrival, intensity or burn probability; basemap; terrain; labels.", "Play or pause.", "Timeline: drag to any moment.", "Playback speed.", "Chart of burned area and spread rate.", "Follow fire: the camera travels with the head of the fire.", "Export a PNG or a WebM video."]))
A(table(["Do this", "To get this"], [["Drag with the left mouse button", "Move the map."], ["Right-drag, or Ctrl + drag", "Rotate and tilt."], ["Click the map", "A probe: arrival time, intensity, burn probability and elevation at that point."], ["Flame size slider", "1&times; is true scale (flame length from the intensity); larger values make small fires visible."]]))
A(sec("5.6", "The Runs tab", "s56"))
A(shot("runs_tab", "The Runs tab.", ["Map layers of the open run: time of arrival, fire intensity, burn probability.", "PDF report of the run.", "Time-lapse video (MP4).", "Compare two runs.", "History: open or delete earlier runs."]))
A(sec("5.7", "The Impact tab", "s57"))
A(shot("impact_tab", "The Impact tab after <b>Show exposure</b>.", ["Show exposure of the open run.", "People, houses, informal settlements, facilities, with the time the fire reaches each.", "Warning time for evacuation.", "Show evacuation routes."]))
A(note("Evacuation needs the road database. If it is not reachable the panel says so; exposure still works."))
A(sec("5.8", "Ensemble and burn probability", "s58"))
A("<p>In the Fire tab, below the run button, <b>Run ensemble</b> repeats the run 12, 24 or 48 times with the wind speed, wind direction, fuel moisture, vegetation greenness and spread rate varied within widths fitted to observed fires. The map then shows the share of runs that burn each cell, and the panel gives the range of burned area.</p>")
A(sec("5.9", "The Mitigation tab", "s59"))
A(shot("mitigate_tab", "The Mitigation tab.", ["Draw a fuel break line (at least three cells wide; only embers cross it).", "Suggest a break across the last run's spread path.", "Draw a treated area.", "Effect of the treatment: remove fuel, halve the spread rate, or cut it to 25 %.", "Run with and without the interventions and compare."]))
A(warn("The effect of a break or treatment is what you set. PAKFIRE does not know how well a real treatment performs."))

A(chap(6, "Discover and monitor", "c6"))
A(shot("discover_tab", "The Discover tab.", ["Rank districts by detections or fire radiative power.", "Districts: <b>Use</b> one as the area.", "Hotspots: strongest or most recent detections.", "Monitoring radius.", "Watch a place."]))
A("<p><b>Live monitoring</b> checks NASA's satellite feed around a point every 1 to 24 hours. When a new detection appears within the radius it runs a 24-hour scenario from the strongest one and can post a message to a webhook (Slack, Discord or ntfy). It is experimental. Detections are 375 m pixels and include flares, kilns and small burns.</p>")

A(chap(7, "Understanding your results", "c7"))
A(sec("7.1", "Arrival, intensity and flame length", "s71"))
A(f'<figure>{ARRIVAL}<figcaption>Time of arrival and intensity, and what intensity means for crews.</figcaption></figure>')
A(table(["Result", "Meaning"], [["Time of arrival", "When the fire front reaches each cell, in hours after the ignition."], ["Fireline intensity", "Heat released per metre of fire front (kW/m). Flame length follows from it."], ["Head spread rate", "How fast the front moves where it moves fastest (m/min)."], ["Burn probability", "Share of ensemble runs that burn the cell."], ["Spot fires", "New fires started by embers ahead of the front."], ["Crown fire area", "Where the fire climbed into the pine canopy."]]))
A(sec("7.2", "How far to trust the numbers", "s72"))
A("<p>On 527 satellite-observed fires held back from calibration (2024-2026) the median burned area was within about a factor of two of the satellite value, with a tendency to over-predict. The overlap of the simulated and the observed burn was 0.30 on average, against 0.29 for a circle of the same area. Fires above 2000 m are over-predicted most. The full record is in the Technical Reference, Chapter 23.</p>")

A(chap(8, "Reports, exports and sharing", "c8"))
A(table(["From", "You get"], [["Runs tab: PDF report", "Map, statistics, weather, warnings and method of the run."], ["Runs tab: Batch PDF", "All saved runs in one report."], ["Runs tab: Time-lapse", "An MP4 of the fire over a terrain background."], ["Runs tab: Spread contours", "Hourly fire fronts as GeoJSON for GIS."], ["Runs tab: Score against an observed perimeter", "Overlap (IoU, Dice), omission, commission and area error against a burned area you upload."], ["3D view: Export", "PNG of the view or a WebM video of a replay."], ["Fire tab: Link", "An address that restores the ignition and weather."]]))

A(chap(9, "Troubleshooting and questions", "c9"))
A(sec("9.1", "Messages you may see", "s91"))
A(table(["Message", "What to do"], [["Ignition is outside Pakistan's data coverage", "Place it inside Pakistan."], ["No burnable fuel within 600 m of this point", "Move the ignition onto vegetation or crops."], ["The fire did not spread", "Fuel too wet or sparse for this weather: try drier or windier weather, or another place."], ["Weather service unavailable", "Choose <b>Set the weather yourself</b> and type it."], ["The ignition lies inside a fuel break or treated area", "Move the ignition or the intervention."], ["Road database unavailable", "Evacuation needs the road server; exposure still works."]]))
A(sec("9.2", "Frequently asked questions", "s92"))
A("<p><b>Why is my forest fire so small?</b> Low-elevation broadleaf forest and moist fuel spread slowly in Rothermel's model; that matches most observed fires. Try the Extreme scenario.</p><p><b>Why does the result change when I run again?</b> Embers land at random. The seed fixes it; the ensemble shows the spread of outcomes.</p><p><b>Can I run outside Pakistan?</b> No: the fuel, terrain and population data cover Pakistan only.</p>")

A(chap(10, "Glossary and data sources", "c10"))
A(table(["Term", "Meaning"], [["Rothermel model", "The standard equations for the spread rate of a surface fire from fuel, moisture, wind and slope."], ["Minimum travel time", "The method that turns spread rates into arrival times over the map."], ["Spotting", "Embers carried by the wind that start new fires ahead of the front."], ["Crown fire", "Fire that burns through the tree canopy, faster and hotter than a surface fire."], ["Dead fuel moisture", "Water content of dead grass, needles and twigs, which follows the weather within hours."], ["FIRMS / VIIRS", "NASA's satellite fire detections, 375 m pixels, a few overpasses a day."]]))
A(table(["Data", "Source"], [["Terrain", "SRTM 30 m"], ["Fuel", "Esri 10 m land cover, Copernicus forest type, MODIS vegetation greenness"], ["Weather", "Open-Meteo, MET Norway"], ["Fire detections", "NASA FIRMS VIIRS"], ["People and houses", "WorldPop 2020, Pakistan Bureau of Statistics 2023 census, VHR-GSIS informal settlements"], ["Roads and fire stations", "OpenStreetMap"]]))
A('<div class="back"><div class="brand">PAKFIRE</div><p>Pakistan Wildfire Spread Simulation, Exposure &amp; Response Analytics<br>AI Geo Navigators, Islamabad</p></div>')

toc = '<section class="toc"><h1>Contents</h1><ul>' + "".join(f'<li class="l{l}"><a href="#{k}">{html.escape(t)}</a></li>' for l, t, k in TOC) + "</ul></section>"
doc = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PAKFIRE User Manual</title><style>{open(os.path.join(HERE, "manual.css")).read()}</style></head><body>' + "".join(body).replace("<!--TOC-->", toc) + "</body></html>"
out = os.path.join(ROOT, "docs", "PAKFIRE_User_Manual.html")
open(out, "w", encoding="utf-8").write(doc)
import weasyprint
weasyprint.HTML(out).write_pdf(out[:-5] + ".pdf")
print("written", out[:-5] + ".pdf")
