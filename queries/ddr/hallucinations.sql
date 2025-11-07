-- Hallucinations: Records in synthetic that do NOT exist in population
-- Formula: S \ P
-- Meaning: Fabricated records that are not factual

SELECT *
FROM synthetic
EXCEPT
SELECT * FROM population;
