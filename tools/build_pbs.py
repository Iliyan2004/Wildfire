"""Builds data/external/pbs_districts.json from the PBS 2023 census district tables on the shared drive: population, households, housing
construction class (pakka / semi-pakka / kacha) and cooking with firewood, plus the WorldPop 2020 total inside each district polygon so that
WorldPop can be scaled to the census count.   python tools/build_pbs.py"""
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkfire import exposure, geo

P = f"{exposure.SHARE}/PBS DATA district/csvs/"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external", "pbs_districts.json")


def main():
    a = pd.read_csv(P + "Area, population by sex, sex ratio, population density, urban proportion, household size and annual growth rate.csv")
    h = pd.read_csv(P + "type of housing unit by region.csv")
    f = pd.read_csv(P + "Housing characteristics, facilities used for fuel, lighting and kitchen by region.csv")
    a, h, f = (x[x.province.notna()].drop_duplicates("name").set_index("name") for x in (a, h, f))     # the unnamed duplicate rows are sub-areas
    D = geo.districts()
    out = {}
    with rasterio.open(exposure.POP) as d:
        for r in D.itertuples():
            if r.name not in a.index:
                continue
            win = from_bounds(*r.geometry.bounds, d.transform).round_offsets().round_lengths()
            arr = d.read(1, window=win, masked=True).filled(0).astype(np.float64)
            arr[arr < 0] = 0
            m = ~geometry_mask([r.geometry], out_shape=arr.shape, transform=rasterio.windows.transform(win, d.transform))
            wp = float(arr[m].sum())
            pop = float(a.loc[r.name, "Pop_2023_Total"])
            hh = float(h.loc[r.name, "Households_AllLocalities"])
            out[r.name] = dict(province=r.province, pop_2023=int(pop), pop_urban=int(a.loc[r.name, "Pop_2023_Urban"]), pop_rural=int(a.loc[r.name, "Pop_2023_Rural"]),
                               area_km2=float(a.loc[r.name, "Area_sq_km"]), growth_pct=float(a.loc[r.name, "Growth_Rate(%)"]), households=int(hh),
                               pakka_pct=round(100 * h.loc[r.name, "Pakka_HH_AllLocalities"] / hh, 1), semi_pakka_pct=round(100 * h.loc[r.name, "Semi_Pakka_HH_AllLocalities"] / hh, 1),
                               kacha_pct=round(100 * h.loc[r.name, "Kacha_HH_AllLocalities"] / hh, 1),
                               firewood_cooking_pct=round(100 * f.loc[r.name, "Fuel_Cooking_Firewood_AllLocalities"] / hh, 1),
                               worldpop_2020=round(wp), census_over_worldpop=round(pop / wp, 3) if wp > 0 else None)
    json.dump(dict(source="PBS 2023 census district tables (shared drive 'PBS DATA district'); WorldPop 2020 UN-adjusted 1 km summed inside each district polygon",
                   districts=out), open(OUT, "w"), indent=1)
    r = np.array([v["census_over_worldpop"] for v in out.values() if v["census_over_worldpop"]])
    print(len(out), "districts; census / WorldPop: median", np.median(r).round(3), "p10", np.percentile(r, 10).round(3), "p90", np.percentile(r, 90).round(3))
    bad = {k: (v["census_over_worldpop"], v["pop_2023"]) for k, v in out.items() if not v["census_over_worldpop"] or not 0.6 < v["census_over_worldpop"] < 1.8}
    print("outside 0.6-1.8:", bad)


if __name__ == "__main__":
    main()
