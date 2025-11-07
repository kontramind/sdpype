-- Duplicate Analysis: Find records that appear multiple times in synthetic data
-- Returns: Unique records with their occurrence count (only duplicated ones)

SELECT *, COUNT(*) as occurrence_count
FROM synthetic
GROUP BY ALL
HAVING COUNT(*) > 1
ORDER BY occurrence_count DESC;
