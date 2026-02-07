SELECT
    year,
    emissions,
    ROUND(
        (emissions - LAG(emissions) OVER (ORDER BY year)) /
        LAG(emissions) OVER (ORDER BY year) * 100, 2
    ) AS growth_rate_pct
FROM global_emissions
WHERE year > (SELECT MIN(year) FROM global_emissions);
