-- Population Matches: Records in synthetic that exist somewhere in population
-- Formula: S ∩ P
-- Meaning: Factual records (includes both DDR and training copies)

SELECT *
FROM synthetic
INTERSECT
SELECT * FROM population;
