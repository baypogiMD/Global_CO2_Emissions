SELECT
    year,
    emissions,
    emissions - LAG(emissions) OVER (ORDER BY year) AS annual_change
FROM global_emissions
ORDER BY year;
