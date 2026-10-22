-- NOAA GHCN-Daily, NYC Central Park station USW00094728 (CC0). by_station CSV has a header row.
CREATE LIGHTNING DATABASE NOAA_GHCN_S3 DESCRIBE BY "NOAA GHCN-Daily S3 filestore source";

CREATE LIGHTNING FILESTORE TABLE ghcn_nyc_central_park FROM NOAA_GHCN_S3 FORMAT CSV OPTIONS (
  PATH "s3a://noaa-ghcn-pds/csv/by_station/USW00094728.csv",
  inferSchema "true",
  header "true",
  isS3BucketPublic "true",
  useS3PathStyleAccess "true",
  s3Endpoint "s3.us-east-1.amazonaws.com"
);

CACHE TABLE NOAA_GHCN_S3.ghcn_nyc_central_park;
