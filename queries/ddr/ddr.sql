-- DDR (Desirable Diverse Records): Records that are factual AND novel
-- Formula: (S ∩ P) \ T
-- Meaning: Records in synthetic that exist in population but NOT in training

-- Returns: Table with all DDR records
SELECT s.*
FROM (
    -- S ∩ P: Records in both synthetic and population (factual)
    SELECT * FROM synthetic
    INTERSECT
    SELECT * FROM population
) AS s
EXCEPT
-- Remove training copies
SELECT * FROM training;
