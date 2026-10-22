#!/usr/bin/env python3
"""Download the small public sample used ONLY by the DEV-ONLY DuckDB source (never the demo). ~140 MB, no login.

Sources and licences:
- NYC TLC yellow taxi trip records, Jan-Feb 2024 (NYC TLC open data, public)
- NOAA GHCN-Daily by-station CSV for Central Park, LaGuardia, JFK (CC0, registry.opendata.aws/noaa-ghcn)
- Nager.Date US public holidays 2024 (MIT-licensed API, date.nager.at)
"""
import urllib.request
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "dev"
FILES = {
    "yellow_tripdata_2024-01.parquet": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
    "yellow_tripdata_2024-02.parquet": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet",
    "taxi_zone_lookup.csv": "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
    "USW00094728.csv": "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/USW00094728.csv",
    "USW00014732.csv": "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/USW00014732.csv",
    "USW00094789.csv": "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/USW00094789.csv",
    "holidays_2024.json": "https://date.nager.at/api/v3/PublicHolidays/2024/US",
}
D.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    p = D / name
    if p.exists() and p.stat().st_size > 0:
        print("have", name)
        continue
    print("fetch", name)
    req = urllib.request.Request(url, headers={"User-Agent": "whybot-dev-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
        f.write(r.read())
print("done:", D)
