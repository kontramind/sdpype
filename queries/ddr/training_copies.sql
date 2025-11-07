-- Training Copies: Records in synthetic that exactly match training data
-- Formula: S ∩ T
-- Meaning: Privacy risk - model memorized training data

SELECT *
FROM synthetic
INTERSECT
SELECT * FROM training;
