-- SchemaStore views that the agents query. The daily taxi aggregate is cached because
-- an uncached yellow_taxi scan reads about 38.7M CSV rows from S3 (around 170 s).
CREATE SCHEMASTORE CONTAINER whybot;

CREATE SCHEMASTORE VIEW yellow_daily WITH CONTAINER whybot AS
SELECT to_date(substr(tpep_pickup_datetime, 1, 10)) AS trip_date,
       count(*) AS trips,
       avg(cast(total_amount AS double)) AS avg_total_amount,
       avg(cast(trip_distance AS double)) AS avg_trip_distance
FROM NY_TAXI.yellow_taxi
WHERE substr(tpep_pickup_datetime, 1, 10) BETWEEN '2024-04-01' AND '2025-02-28'
GROUP BY to_date(substr(tpep_pickup_datetime, 1, 10));

CREATE SCHEMASTORE VIEW nyc_weather_daily WITH CONTAINER whybot AS
SELECT to_date(cast(`DATE` AS string), 'yyyyMMdd') AS obs_date,
       max(CASE WHEN ELEMENT = 'PRCP' THEN DATA_VALUE END) / 10.0 AS prcp_mm,
       max(CASE WHEN ELEMENT = 'SNOW' THEN DATA_VALUE END) AS snow_mm,
       max(CASE WHEN ELEMENT = 'TMAX' THEN DATA_VALUE END) / 10.0 AS tmax_c,
       max(CASE WHEN ELEMENT = 'TMIN' THEN DATA_VALUE END) / 10.0 AS tmin_c
FROM NOAA_GHCN_S3.ghcn_nyc_central_park
WHERE `DATE` >= 20240101
GROUP BY to_date(cast(`DATE` AS string), 'yyyyMMdd');

CREATE SCHEMASTORE VIEW us_holidays WITH CONTAINER whybot AS
SELECT to_date(`date`) AS holiday_date, name AS holiday_name, `global` AS is_national
FROM HOLIDAYS_REST.us_holidays_2024
UNION ALL
SELECT to_date(`date`) AS holiday_date, name AS holiday_name, `global` AS is_national
FROM HOLIDAYS_REST.us_holidays_2025;

CACHE TABLE whybot.yellow_daily;
