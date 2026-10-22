-- Federated daily view: NY_TAXI yellow trips (S3 CSV) x NOAA Central Park weather (public S3 CSV)
-- x Nager.Date US holidays (REST) x TPCDS_DB.date_dim (Parquet: day name, weekend and d_holiday flags).
-- Note: TPCDS d_holiday is synthetic (3 days a year, e.g. 2024-07-05 and 2024-12-26); use h.holiday_name for real holidays.
SELECT t.trip_date,
       dd.d_day_name,
       dd.d_weekend,
       dd.d_holiday AS tpcds_holiday_flag,
       h.holiday_name,
       t.trips,
       round(t.avg_total_amount, 2) AS avg_total_amount,
       w.prcp_mm,
       w.snow_mm,
       w.tmax_c,
       w.tmin_c
FROM whybot.yellow_daily t
JOIN whybot.nyc_weather_daily w ON w.obs_date = t.trip_date
JOIN TPCDS_DB.date_dim dd ON dd.d_date = t.trip_date
LEFT JOIN whybot.us_holidays h ON h.holiday_date = t.trip_date AND h.is_national = true
WHERE t.trip_date BETWEEN DATE '2024-11-15' AND DATE '2025-01-15'
ORDER BY t.trip_date;
