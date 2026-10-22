-- Re-run after any driver restart (docker compose restart/down/up): Spark caches are in memory.
-- The yellow_daily cache build scans about 38.7M rows from S3 and takes around 3 minutes on 2 cores.
CACHE TABLE NOAA_GHCN_S3.ghcn_nyc_central_park;

CACHE TABLE whybot.yellow_daily;
