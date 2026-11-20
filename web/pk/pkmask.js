// Pakistan-only map, as in SAILAB: everything outside the country is darkened, labels outside fade, and the map cannot be panned away.
// Usage: pkMask(map) after the map is created. The outline comes from the district boundaries (/api/pakistan.geojson).
function pkMask(map) {
  const BOUNDS = [[58.5, 21.5], [80.5, 38.8]];
  map.setMaxBounds(BOUNDS);
  map.setMinZoom(Math.max(map.getMinZoom(), 3.6));
  let data = null;
  const add = () => {
    if (!data || map.getSource("pk-mask")) return;
    map.addSource("pk-mask", { type: "geojson", data });
    const before = map.getStyle().layers.find((l) => l.type === "symbol");
    map.addLayer({ id: "pk-mask", type: "fill", source: "pk-mask", filter: ["==", ["get", "kind"], "mask"], paint: { "fill-color": "#05070a", "fill-opacity": 0.72 } }, before && before.id);
    map.addLayer({ id: "pk-outline", type: "line", source: "pk-mask", filter: ["==", ["get", "kind"], "outline"], paint: { "line-color": "#FF6A2B", "line-opacity": 0.55, "line-width": 1.2 } });
  };
  fetch("/api/pakistan.geojson").then((r) => r.json()).then((j) => { data = j; if (map.isStyleLoaded()) add(); }).catch(() => {});
  map.on("style.load", add);
  map.on("styledata", add);
}
