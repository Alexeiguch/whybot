-- Nager.Date US public holidays (https://date.nager.at, MIT licence, no key).
-- The response is a top-level JSON array; Zetaris 2.4.3.1 accepts it (confirmed 22 Oct 2026).
CREATE LIGHTNING DATABASE HOLIDAYS_REST DESCRIBE BY "Nager.Date public holidays REST source";

CREATE LIGHTNING REST TABLE us_holidays_2024 FROM HOLIDAYS_REST REQUEST(
    endpoint "https://date.nager.at/api/v3/PublicHolidays/2024/US",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE LIGHTNING REST TABLE us_holidays_2025 FROM HOLIDAYS_REST REQUEST(
    endpoint "https://date.nager.at/api/v3/PublicHolidays/2025/US",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();
