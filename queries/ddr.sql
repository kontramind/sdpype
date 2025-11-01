-- ============================================================================
-- DDR Metric Queries for DuckDB
-- ============================================================================
-- This file contains all SQL queries for computing DDR (Desirable Diverse
-- Records) metrics. Queries are separated by comment markers: -- @query: name
--
-- Usage: Load this file and execute queries by name
-- ============================================================================

-- @query: summary
-- Compute all DDR metrics in a single query - dual perspective (unique + total)
-- Returns: Single row with all counts and rates
-- Note: {{HASH_COLS}} is replaced by Python with actual column names
WITH
-- Hash all rows for efficient matching
synthetic_hashed AS (
    SELECT *, hash({{HASH_COLS}}) as row_hash FROM synthetic
),
population_hashes AS (
    SELECT DISTINCT hash({{HASH_COLS}}) as row_hash FROM population
),
training_hashes AS (
    SELECT DISTINCT hash({{HASH_COLS}}) as row_hash FROM training
),

-- Count total synthetic records (all rows including duplicates)
synthetic_total AS (
    SELECT COUNT(*) as total FROM synthetic_hashed
),

-- Count unique synthetic records
synthetic_unique AS (
    SELECT COUNT(DISTINCT row_hash) as count FROM synthetic_hashed
),

-- TOTAL COUNTS (including duplicates) --

-- DDR (total): in population AND not in training
ddr_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
),

-- Hallucinations (total): not in population
hallucination_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
),

-- Training copies (total): in training
training_copy_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
),

-- Population matches (total): in population
population_match_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
),

-- UNIQUE COUNTS (distinct records only) --

-- DDR (unique): distinct hashes in population AND not in training
ddr_unique AS (
    SELECT COUNT(DISTINCT s.row_hash) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
),

-- Hallucinations (unique): distinct hashes not in population
hallucination_unique AS (
    SELECT COUNT(DISTINCT s.row_hash) as count
    FROM synthetic_hashed s
    WHERE s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
),

-- Training copies (unique): distinct hashes in training
training_copy_unique AS (
    SELECT COUNT(DISTINCT s.row_hash) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
),

-- Population matches (unique): distinct hashes in population
population_match_unique AS (
    SELECT COUNT(DISTINCT s.row_hash) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
)

-- Compute final metrics
SELECT
    -- Basic counts
    st.total as total_synthetic_records,
    su.count as unique_synthetic_records,
    (st.total - su.count) as duplicate_records,
    ROUND(((st.total - su.count)::FLOAT / st.total * 100), 2) as duplicate_rate_pct,

    -- UNIQUE metrics (distinct records)
    ddr_u.count as ddr_unique_count,
    ROUND((ddr_u.count::FLOAT / su.count * 100), 2) as ddr_unique_rate_pct,

    hall_u.count as hallucination_unique_count,
    ROUND((hall_u.count::FLOAT / su.count * 100), 2) as hallucination_unique_rate_pct,

    train_u.count as training_copy_unique_count,
    ROUND((train_u.count::FLOAT / su.count * 100), 2) as training_copy_unique_rate_pct,

    pop_u.count as population_match_unique_count,
    ROUND((pop_u.count::FLOAT / su.count * 100), 2) as population_match_unique_rate_pct,

    -- TOTAL metrics (including duplicates)
    ddr_t.count as ddr_total_count,
    ROUND((ddr_t.count::FLOAT / st.total * 100), 2) as ddr_total_rate_pct,

    hall_t.count as hallucination_total_count,
    ROUND((hall_t.count::FLOAT / st.total * 100), 2) as hallucination_total_rate_pct,

    train_t.count as training_copy_total_count,
    ROUND((train_t.count::FLOAT / st.total * 100), 2) as training_copy_total_rate_pct,

    pop_t.count as population_match_total_count,
    ROUND((pop_t.count::FLOAT / st.total * 100), 2) as population_match_total_rate_pct

FROM synthetic_total st
CROSS JOIN synthetic_unique su
CROSS JOIN ddr_unique ddr_u
CROSS JOIN hallucination_unique hall_u
CROSS JOIN training_copy_unique train_u
CROSS JOIN population_match_unique pop_u
CROSS JOIN ddr_total ddr_t
CROSS JOIN hallucination_total hall_t
CROSS JOIN training_copy_total train_t
CROSS JOIN population_match_total pop_t;

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
