-- ============================================================================
-- DDR Metric Queries for DuckDB
-- ============================================================================
-- This file contains all SQL queries for computing DDR (Desirable Diverse
-- Records) metrics. Queries are separated by comment markers: -- @query: name
--
-- Usage: Load this file and execute queries by name
-- ============================================================================

-- @query: summary
-- Compute all DDR metrics in a single query
-- Returns: Single row with all counts and rates
WITH
-- Count total synthetic records
synthetic_total AS (
    SELECT COUNT(*) as total FROM synthetic
),
-- Count DDR records: (S ∩ P) \ T
ddr_records AS (
    SELECT COUNT(*) as count FROM (
        SELECT * FROM (
            SELECT * FROM synthetic
            INTERSECT
            SELECT * FROM population
        )
        EXCEPT
        SELECT * FROM training
    )
),
-- Count hallucinations: S \ P
hallucination_records AS (
    SELECT COUNT(*) as count FROM (
        SELECT * FROM synthetic
        EXCEPT
        SELECT * FROM population
    )
),
-- Count training copies: S ∩ T
training_copy_records AS (
    SELECT COUNT(*) as count FROM (
        SELECT * FROM synthetic
        INTERSECT
        SELECT * FROM training
    )
),
-- Count population matches: S ∩ P
population_match_records AS (
    SELECT COUNT(*) as count FROM (
        SELECT * FROM synthetic
        INTERSECT
        SELECT * FROM population
    )
),
-- Count unique synthetic records
synthetic_unique AS (
    SELECT COUNT(*) as count FROM (
        SELECT DISTINCT * FROM synthetic
    )
)

-- Compute final metrics
SELECT
    st.total as total_synthetic_records,
    su.count as unique_synthetic_records,
    (st.total - su.count) as duplicate_records,
    ROUND(((st.total - su.count)::FLOAT / st.total * 100), 2) as duplicate_rate_pct,

    ddr.count as ddr_count,
    ROUND((ddr.count::FLOAT / st.total * 100), 2) as ddr_rate_pct,

    hall.count as hallucination_count,
    ROUND((hall.count::FLOAT / st.total * 100), 2) as hallucination_rate_pct,

    train.count as training_copy_count,
    ROUND((train.count::FLOAT / st.total * 100), 2) as training_copy_rate_pct,

    pop.count as population_match_count,
    ROUND((pop.count::FLOAT / st.total * 100), 2) as population_match_rate_pct

FROM synthetic_total st
CROSS JOIN ddr_records ddr
CROSS JOIN hallucination_records hall
CROSS JOIN training_copy_records train
CROSS JOIN population_match_records pop
CROSS JOIN synthetic_unique su;

-- @query: ddr
-- DDR (Desirable Diverse Records): Records that are factual AND novel
-- Formula: (S ∩ P) \ T
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

-- @query: hallucinations
-- Hallucinations: Records in synthetic that do NOT exist in population
-- Formula: S \ P
-- Returns: Table with all hallucinated records
SELECT *
FROM synthetic
EXCEPT
SELECT * FROM population;

-- @query: training_copies
-- Training Copies: Records in synthetic that exactly match training data
-- Formula: S ∩ T
-- Returns: Table with all training copy records
SELECT *
FROM synthetic
INTERSECT
SELECT * FROM training;

-- @query: population_matches
-- Population Matches: Records in synthetic that exist somewhere in population
-- Formula: S ∩ P
-- Returns: Table with all factual records (includes both DDR and training copies)
SELECT *
FROM synthetic
INTERSECT
SELECT * FROM population;

-- @query: duplicates
-- Duplicate Analysis: Find records that appear multiple times in synthetic data
-- Returns: Unique records with their occurrence count (only duplicated ones)
SELECT *, COUNT(*) as occurrence_count
FROM synthetic
GROUP BY ALL
HAVING COUNT(*) > 1
ORDER BY occurrence_count DESC;
