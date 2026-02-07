SELECT
    COUNT(*) AS total_rows,
    MIN(year) AS min_year,
    MAX(year) AS max_year,
    MIN(emissions) AS min_emissions,
    MAX(emissions) AS max_emissions
FROM global_emissions;
