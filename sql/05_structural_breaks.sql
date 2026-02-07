SELECT
    CASE
        WHEN year < 1850 THEN 'Pre-Industrial'
        WHEN year BETWEEN 1850 AND 1949 THEN 'Industrialization'
        ELSE 'Post-1950 Acceleration'
    END AS era,
    AVG(emissions) AS avg_emissions
FROM global_emissions
GROUP BY era;
