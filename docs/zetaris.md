# Zetaris in whybot

## Instance
- Local Docker Compose install of Zetaris 2.4.3.1 (build cb88df0b). It lives in `/workspace/hackathon/zetaris-platform` and runs postgres, driver, api and ui, with init and bootstrap exiting 0.
- `sudo ./verify.sh` passes 9 of 9. Telemetry is declined with `TELEMETRY_ENABLED=false` in `.env`.
- JDBC: `jdbc:zetaris:lightning@localhost:10000`, class `com.zetaris.lightning.jdbc.LightningDriver`, JAR `zetaris-platform/jdbc/ndp-jdbc-driver-2.4.3.1-7eff043-driver.jar`.
- Credentials live in `/workspace/hackathon/.env.local` (mode 600, gitignored), as `ZETARIS_USERNAME` and `ZETARIS_PASSWORD`. Never log them.
- REST needs `ZETARIS_API_KEY` (created in the GUI) and `ZETARIS_ORG_ID`. Neither is set yet; JDBC is the working channel.
- Python access: jaydebeapi with JPype and a JRE (Java 21 on the box). See `zetaris-getting-started/scripts/zetaris_sql.py` `JdbcChannel` for a working client.

## Box quirks (not part of the product)
- dockerd runs by hand: `sudo dockerd > /tmp/dockerd.log 2>&1 &`. There is no systemd.
- The box's legacy iptables has a FORWARD DROP policy that blocked traffic on the compose bridge. Fix it with:
  `sudo iptables-legacy -I DOCKER-USER -i br-93fc3ab2c5f9 -j ACCEPT && sudo iptables-legacy -I DOCKER-USER -o br-93fc3ab2c5f9 -j ACCEPT`.
  This does not persist. Re-apply it after a dockerd restart; the bridge name comes from `docker network ls`, the `zetaris_default` network.

## Data coverage (measured 22 Oct 2026 over JDBC)
| Source | Table | Rows | Date range | Notes |
|---|---|---|---|---|
| NY_TAXI (S3 CSV) | yellow_taxi | 38,667,711 | 2024-04 to 2025-02 (min and max show outliers from 2002 and 2026) | Real TLC yellow trips, about 3.0 to 3.8M a month. An uncached scan takes about 170 s |
| NY_TAXI | hvv | 218,911,709 | 2024-04-01 to 2025-02-28 | High-volume FHV (Uber/Lyft). A full scan takes about 24 min, so aggregate and cache before use |
| NY_TAXI | for_hire_vehicles | 32,835,497 | 2022-01 to 2022-09 | |
| NY_TAXI | green_taxi | 1,272,706 | 2022-01 to 2022-09 (636k rows with null dates) | |
| NY_TAXI | taxi_trips | 500,000 | 2024-07-27 to 2024-08-01 | Enriched, derived trips |
| NY_TAXI | trips, weather_data, surge_pricing, driver_shifts, trip_payments | 100,000 each | about 2023-08 or 2024-08 to 2025-08 | **Synthetic.** For example, weather_data has 10,000 zone ids and implausible values. Do not use it as real evidence |
| NY_TAXI | customers, drivers, vehicles, zones | 100,000 / 100,000 / 100,000 / 10,000 | | Synthetic dimensions |
| NY_TAXI | taxi_zones | 789 | | Real TLC zone polygons |
| TPCDS_DB (S3 Parquet) | date_dim and 23 others | date_dim has 73,049 | 1900-01-02 to 2100-01-01 | `d_holiday` is synthetic (650 Y in total: 3 a year, inconsistent, e.g. 2022-07-05 and 2024-12-31). `d_weekend` is Y on Fri and Sat. Use it for d_day_name, not for real holidays |
| NOAA_GHCN_S3 (public S3 CSV, CC0) | ghcn_nyc_central_park | 16 MB station file | 1869 to **2025-02-05** | Station USW00094728. Daily PRCP, SNOW, SNWD, TMAX, TMIN and AWND for 311 days from 2024-04-01. PRCP and TMAX/TMIN are in tenths. Cached |
| HOLIDAYS_REST (Nager.Date REST) | us_holidays_2024, us_holidays_2025 | 34 total | 2024 and 2025 | Real US holidays; `global = true` marks national ones |

The usable overlap for a "why did trips change" question is **2024-04-01 to 2025-02-05**: yellow taxi, hvv, NOAA weather and holidays.

## Views (container `whybot`)
- `whybot.yellow_daily`: one row per day (334 days), with trips, avg_total_amount and avg_trip_distance. It is cached; building the cache takes about 3 min.
- `whybot.nyc_weather_daily`: one row per day from 2024-01-01 to 2025-02-05, with prcp_mm, snow_mm, tmax_c and tmin_c.
- `whybot.us_holidays`: holiday_date, holiday_name and is_national.

## Federated query
`sql/10_federated_daily_select.sql` joins `whybot.yellow_daily` with `whybot.nyc_weather_daily`, `TPCDS_DB.date_dim` and `whybot.us_holidays`. A 62-day window returns in about 17 s.
Sample rows: 2024-11-28 (Thanksgiving) had 77,244 trips against about 104 to 115k on nearby weekdays. 2024-12-25 (Christmas, d_holiday Y) had 54,764. 2024-12-21 had 46 mm of snow.

## Gotchas
- One statement per call, with no trailing semicolon, through `JdbcChannel.run`. `scripts/run_sql.py` splits files on `;`.
- Quote `DATE`, `date` and `global` with backticks in SchemaStore views.
- CACHE TABLE lives in memory and is lost on a driver restart; re-run `sql/04_cache_refresh.sql`.
- A judge cannot reproduce this instance without their own bundle, so whybot needs a recorded sample-mode fallback.
