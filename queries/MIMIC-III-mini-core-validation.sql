-- ============================================================================
-- Unified Validation Queries for DDR and Plausibility Metrics
-- ============================================================================
--
-- DATASET: MIMIC-III-mini-core Dataset
-- This validation file is DATASET-SPECIFIC and contains hardcoded column names
-- and validation rules for the MIMIC-III-mini-core dataset.
--
-- This SQL file contains queries to compute:
-- 1. DDR (Desirable Diverse Records) metrics - records that exist in population
--    but not in training data (using BINNED numerical values)
-- 2. Plausibility metrics - records that pass all validation rules
--
-- Both metrics are computed with dual perspectives:
-- - Unique: Count of distinct records (by hash)
-- - Total: Count of all records including duplicates
--
-- Factuality Approach:
-- - Numerical columns are binned into 20 equal-width bins based on population
-- - NULL values become bin 0
-- - Hash is computed on binned numericals + categorical values
--
-- Plausibility Rules:
-- - Categorical membership: GENDER, ETHGRP, ADMTYPE, READMIT
-- - Numerical ranges: AGE, NTproBNP, CREAT, BUN, POTASS, CHOL, HR, SBP, DBP, RR
--   (within population min/max)
--
-- ============================================================================


-- ============================================================================
-- COLUMN CONFIGURATION
-- ============================================================================
-- To add/remove columns from validation:
-- 1. Update the lists below by commenting/uncommenting lines
-- 2. Update the corresponding sections marked with [UPDATE REQUIRED]
--
-- CATEGORICAL COLUMNS (for categorical validation):
--   - GENDER
--   - ETHGRP
--   - ADMTYPE
--   - READMIT
--
-- NUMERIC COLUMNS (for binning and range validation):
--   - AGE
--   - NTproBNP
--   - CREAT
--   - BUN
--   - POTASS
--   - CHOL
--   - HR
--   - SBP
--   - DBP
--   - RR
--
-- When adding/removing columns, update these sections:
-- [1] num_ranges CTE (lines ~95-110)
-- [2] population_binned CTE (lines ~120-140)
-- [3] training_binned CTE (lines ~145-165)
-- [4] synthetic_binned CTE (lines ~170-190)
-- [5] Hash calculations (lines ~195-240)
-- [6] Valid categorical CTEs (lines ~315-335)
-- [7] Plausibility validation (lines ~345-395)
--
-- ============================================================================

-- @query: summary
--
-- Compute all metrics in a single query for efficiency:
-- - Synthetic dataset statistics (total, unique counts)
-- - DDR metrics (desirable records: in population but not in training)
-- - Hallucination metrics (records not in population)
-- - Training copy metrics (records that match training data)
-- - Plausibility metrics (records passing all validation rules)
--

-- ============================================================================
-- Reusable Binning Macro
-- ============================================================================
-- Creates equal-width bins for numerical columns, returned as strings
-- - NULL values → "Missing"
-- - Valid values → "1" to "num_bins" (as strings)
-- - Handles edge cases: division by zero, values outside range
CREATE OR REPLACE MACRO bin_numeric(val, min_val, max_val, num_bins) AS
    CASE WHEN val IS NULL THEN 'Missing'
         ELSE CAST(LEAST(GREATEST(
             FLOOR((val - min_val) / NULLIF(max_val - min_val, 0) * num_bins) + 1,
             1), num_bins) AS VARCHAR)
    END;

WITH
-- ============================================================================
-- [1] Numerical Column Ranges from Population (for binning)
-- [UPDATE REQUIRED] Add/remove numeric columns here
-- ============================================================================
num_ranges AS (
    SELECT
        -- Add MIN/MAX for each numeric column:
        MIN(AGE) as age_min, MAX(AGE) as age_max,
        MIN(NTproBNP) as ntprobnp_min, MAX(NTproBNP) as ntprobnp_max,
        MIN(CREAT) as creat_min, MAX(CREAT) as creat_max,
        MIN(BUN) as bun_min, MAX(BUN) as bun_max,
        MIN(POTASS) as potass_min, MAX(POTASS) as potass_max,
        MIN(CHOL) as chol_min, MAX(CHOL) as chol_max,
        MIN(HR) as hr_min, MAX(HR) as hr_max,
        MIN(SBP) as sbp_min, MAX(SBP) as sbp_max,
        MIN(DBP) as dbp_min, MAX(DBP) as dbp_max,
        MIN(RR) as rr_min, MAX(RR) as rr_max
        -- To add a column: MIN(COLUMN_NAME) as column_min, MAX(COLUMN_NAME) as column_max,
        -- To remove a column: comment out or delete the corresponding line
    FROM population
),

-- ============================================================================
-- [2] Binned Datasets for Factuality Checks (20 bins as strings)
-- [UPDATE REQUIRED] Add/remove columns here
-- Numerical columns → bin strings: "Missing", "1", "2", ..., "20"
-- ============================================================================
population_binned AS (
    SELECT
        -- [2a] Categorical columns (add/remove as-is, no binning)
        GENDER,
        ETHGRP,
        ADMTYPE,
        READMIT,
        -- [2b] Numerical columns (add/remove bin_numeric calls)
        bin_numeric(AGE, r.age_min, r.age_max, 20) as AGE_BIN,
        bin_numeric(NTproBNP, r.ntprobnp_min, r.ntprobnp_max, 20) as NTPROBNP_BIN,
        bin_numeric(CREAT, r.creat_min, r.creat_max, 20) as CREAT_BIN,
        bin_numeric(BUN, r.bun_min, r.bun_max, 20) as BUN_BIN,
        bin_numeric(POTASS, r.potass_min, r.potass_max, 20) as POTASS_BIN,
        bin_numeric(CHOL, r.chol_min, r.chol_max, 20) as CHOL_BIN,
        bin_numeric(HR, r.hr_min, r.hr_max, 20) as HR_BIN,
        bin_numeric(SBP, r.sbp_min, r.sbp_max, 20) as SBP_BIN,
        bin_numeric(DBP, r.dbp_min, r.dbp_max, 20) as DBP_BIN,
        bin_numeric(RR, r.rr_min, r.rr_max, 20) as RR_BIN
        -- To add: bin_numeric(COLUMN_NAME, r.column_min, r.column_max, 20) as COLUMN_BIN,
    FROM population, num_ranges r
),

-- [3] Training Binned - Same structure as population_binned
-- [UPDATE REQUIRED] Keep in sync with population_binned
training_binned AS (
    SELECT
        -- [3a] Categorical columns (must match population_binned)
        GENDER,
        ETHGRP,
        ADMTYPE,
        READMIT,
        -- [3b] Numerical columns (must match population_binned)
        bin_numeric(AGE, r.age_min, r.age_max, 20) as AGE_BIN,
        bin_numeric(NTproBNP, r.ntprobnp_min, r.ntprobnp_max, 20) as NTPROBNP_BIN,
        bin_numeric(CREAT, r.creat_min, r.creat_max, 20) as CREAT_BIN,
        bin_numeric(BUN, r.bun_min, r.bun_max, 20) as BUN_BIN,
        bin_numeric(POTASS, r.potass_min, r.potass_max, 20) as POTASS_BIN,
        bin_numeric(CHOL, r.chol_min, r.chol_max, 20) as CHOL_BIN,
        bin_numeric(HR, r.hr_min, r.hr_max, 20) as HR_BIN,
        bin_numeric(SBP, r.sbp_min, r.sbp_max, 20) as SBP_BIN,
        bin_numeric(DBP, r.dbp_min, r.dbp_max, 20) as DBP_BIN,
        bin_numeric(RR, r.rr_min, r.rr_max, 20) as RR_BIN
    FROM training, num_ranges r
),

-- [4] Synthetic Binned - Similar to population_binned but with aliases
-- [UPDATE REQUIRED] Keep in sync with population_binned
synthetic_binned AS (
    SELECT
        s.*,
        -- [4a] Categorical columns (create _CAT aliases for hash)
        s.GENDER as GENDER_CAT,
        s.ETHGRP as ETHGRP_CAT,
        s.ADMTYPE as ADMTYPE_CAT,
        s.READMIT as READMIT_CAT,
        -- [4b] Numerical columns (must match population_binned)
        bin_numeric(s.AGE, r.age_min, r.age_max, 20) as AGE_BIN,
        bin_numeric(s.NTproBNP, r.ntprobnp_min, r.ntprobnp_max, 20) as NTPROBNP_BIN,
        bin_numeric(s.CREAT, r.creat_min, r.creat_max, 20) as CREAT_BIN,
        bin_numeric(s.BUN, r.bun_min, r.bun_max, 20) as BUN_BIN,
        bin_numeric(s.POTASS, r.potass_min, r.potass_max, 20) as POTASS_BIN,
        bin_numeric(s.CHOL, r.chol_min, r.chol_max, 20) as CHOL_BIN,
        bin_numeric(s.HR, r.hr_min, r.hr_max, 20) as HR_BIN,
        bin_numeric(s.SBP, r.sbp_min, r.sbp_max, 20) as SBP_BIN,
        bin_numeric(s.DBP, r.dbp_min, r.dbp_max, 20) as DBP_BIN,
        bin_numeric(s.RR, r.rr_min, r.rr_max, 20) as RR_BIN
    FROM synthetic s, num_ranges r
),

-- ============================================================================
-- [5] Hash Datasets for Factuality Comparisons (using binned values)
-- [UPDATE REQUIRED] Add/remove columns in hash() calls
-- ============================================================================
synthetic_hashed AS (
    SELECT
        *,
        -- [5a] Hash: categorical _CAT aliases + _BIN columns (order matters!)
        hash(GENDER_CAT, ETHGRP_CAT, ADMTYPE_CAT, READMIT_CAT,
             AGE_BIN, NTPROBNP_BIN, CREAT_BIN, BUN_BIN, POTASS_BIN, CHOL_BIN, HR_BIN, SBP_BIN, DBP_BIN, RR_BIN) as row_hash
    FROM synthetic_binned
),

population_hashes AS (
    SELECT DISTINCT
        -- [5b] Hash: categorical columns + _BIN columns (must match order of 5a)
        hash(GENDER, ETHGRP, ADMTYPE, READMIT,
             AGE_BIN, NTPROBNP_BIN, CREAT_BIN, BUN_BIN, POTASS_BIN, CHOL_BIN, HR_BIN, SBP_BIN, DBP_BIN, RR_BIN) as row_hash
    FROM population_binned
),

training_hashes AS (
    SELECT DISTINCT
        -- [5c] Hash: categorical columns + _BIN columns (must match order of 5a)
        hash(GENDER, ETHGRP, ADMTYPE, READMIT,
             AGE_BIN, NTPROBNP_BIN, CREAT_BIN, BUN_BIN, POTASS_BIN, CHOL_BIN, HR_BIN, SBP_BIN, DBP_BIN, RR_BIN) as row_hash
    FROM training_binned
),

-- ============================================================================
-- [5d] Dataset Statistics (also uses hash calculations)
-- [UPDATE REQUIRED] Keep hash columns in sync with [5a-5c]
-- ============================================================================
population_stats AS (
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT row_hash) as unique_count
    FROM (SELECT hash(GENDER, ETHGRP, ADMTYPE, READMIT,
             AGE_BIN, NTPROBNP_BIN, CREAT_BIN, BUN_BIN, POTASS_BIN, CHOL_BIN, HR_BIN, SBP_BIN, DBP_BIN, RR_BIN) as row_hash
          FROM population_binned)
),

training_stats AS (
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT row_hash) as unique_count
    FROM (SELECT hash(GENDER, ETHGRP, ADMTYPE, READMIT,
             AGE_BIN, NTPROBNP_BIN, CREAT_BIN, BUN_BIN, POTASS_BIN, CHOL_BIN, HR_BIN, SBP_BIN, DBP_BIN, RR_BIN) as row_hash
          FROM training_binned)
),

synthetic_stats AS (
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT row_hash) as unique_count
    FROM synthetic_hashed
),

-- ============================================================================
-- DDR Metrics: Records in population but not in training
-- ============================================================================
ddr_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
),

ddr_unique AS (
    SELECT COUNT(DISTINCT s.row_hash) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
),

-- ============================================================================
-- Training Copy Metrics - Split into Valid vs Propagation
-- ============================================================================

-- Training Copy (Valid): Copied from training AND exists in population (real data)
training_copy_valid_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
      AND s.row_hash IN (SELECT row_hash FROM population_hashes)
),

training_copy_valid_unique AS (
    SELECT COUNT(DISTINCT s.row_hash) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
      AND s.row_hash IN (SELECT row_hash FROM population_hashes)
),

-- Training Copy (Propagation): Copied from training but NOT in population (hallucination propagation!)
training_copy_propagation_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
),

training_copy_propagation_unique AS (
    SELECT COUNT(DISTINCT s.row_hash) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
),

-- ============================================================================
-- Hallucination Metrics - New Hallucinations Only
-- ============================================================================

-- New Hallucinations: NOT in population AND NOT in training (freshly fabricated)
new_hallucination_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
),

new_hallucination_unique AS (
    SELECT COUNT(DISTINCT s.row_hash) as count
    FROM synthetic_hashed s
    WHERE s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
),

-- ============================================================================
-- [6] Plausibility Rules: Extract valid values/ranges from population
-- [UPDATE REQUIRED] Add/remove valid categorical value CTEs
-- ============================================================================

-- [6a] Valid categorical values from population (one CTE per categorical column)
valid_genders AS (
    SELECT DISTINCT GENDER as value FROM population
),

valid_ethgrps AS (
    SELECT DISTINCT ETHGRP as value FROM population
),

valid_admtypes AS (
    SELECT DISTINCT ADMTYPE as value FROM population
),

valid_readmits AS (
    SELECT DISTINCT READMIT as value FROM population
),
-- To add a categorical column: Add a CTE like:
-- valid_column_name AS (
--     SELECT DISTINCT COLUMN_NAME as value FROM population
-- ),

-- ============================================================================
-- [7] Plausibility Validation: Check each synthetic record against all rules
-- [UPDATE REQUIRED] Add/remove validation checks for each column
-- ============================================================================

synthetic_with_validity AS (
    SELECT
        s.*,
        -- [7a] Check categorical rules (one CASE per categorical column)
        CASE WHEN s.GENDER IN (SELECT value FROM valid_genders) THEN 1 ELSE 0 END as gender_valid,
        CASE WHEN s.ETHGRP IN (SELECT value FROM valid_ethgrps) THEN 1 ELSE 0 END as ethgrp_valid,
        CASE WHEN s.ADMTYPE IN (SELECT value FROM valid_admtypes) THEN 1 ELSE 0 END as admtype_valid,
        CASE WHEN s.READMIT IN (SELECT value FROM valid_readmits) THEN 1 ELSE 0 END as readmit_valid,
        -- [7b] Check numerical range rules (NULL is allowed, one CASE per numeric column)
        CASE WHEN s.AGE IS NULL OR (s.AGE >= (SELECT age_min FROM num_ranges) AND s.AGE <= (SELECT age_max FROM num_ranges)) THEN 1 ELSE 0 END as age_valid,
        CASE WHEN s.NTproBNP IS NULL OR (s.NTproBNP >= (SELECT ntprobnp_min FROM num_ranges) AND s.NTproBNP <= (SELECT ntprobnp_max FROM num_ranges)) THEN 1 ELSE 0 END as ntprobnp_valid,
        CASE WHEN s.CREAT IS NULL OR (s.CREAT >= (SELECT creat_min FROM num_ranges) AND s.CREAT <= (SELECT creat_max FROM num_ranges)) THEN 1 ELSE 0 END as creat_valid,
        CASE WHEN s.BUN IS NULL OR (s.BUN >= (SELECT bun_min FROM num_ranges) AND s.BUN <= (SELECT bun_max FROM num_ranges)) THEN 1 ELSE 0 END as bun_valid,
        CASE WHEN s.POTASS IS NULL OR (s.POTASS >= (SELECT potass_min FROM num_ranges) AND s.POTASS <= (SELECT potass_max FROM num_ranges)) THEN 1 ELSE 0 END as potass_valid,
        CASE WHEN s.CHOL IS NULL OR (s.CHOL >= (SELECT chol_min FROM num_ranges) AND s.CHOL <= (SELECT chol_max FROM num_ranges)) THEN 1 ELSE 0 END as chol_valid,
        CASE WHEN s.HR IS NULL OR (s.HR >= (SELECT hr_min FROM num_ranges) AND s.HR <= (SELECT hr_max FROM num_ranges)) THEN 1 ELSE 0 END as hr_valid,
        CASE WHEN s.SBP IS NULL OR (s.SBP >= (SELECT sbp_min FROM num_ranges) AND s.SBP <= (SELECT sbp_max FROM num_ranges)) THEN 1 ELSE 0 END as sbp_valid,
        CASE WHEN s.DBP IS NULL OR (s.DBP >= (SELECT dbp_min FROM num_ranges) AND s.DBP <= (SELECT dbp_max FROM num_ranges)) THEN 1 ELSE 0 END as dbp_valid,
        CASE WHEN s.RR IS NULL OR (s.RR >= (SELECT rr_min FROM num_ranges) AND s.RR <= (SELECT rr_max FROM num_ranges)) THEN 1 ELSE 0 END as rr_valid,
        -- [7c] Check if ALL rules pass (must include all categorical + numeric checks)
        CASE WHEN
            s.GENDER IN (SELECT value FROM valid_genders)
            AND s.ETHGRP IN (SELECT value FROM valid_ethgrps)
            AND s.ADMTYPE IN (SELECT value FROM valid_admtypes)
            AND s.READMIT IN (SELECT value FROM valid_readmits)
            AND (s.AGE IS NULL OR (s.AGE >= (SELECT age_min FROM num_ranges) AND s.AGE <= (SELECT age_max FROM num_ranges)))
            AND (s.NTproBNP IS NULL OR (s.NTproBNP >= (SELECT ntprobnp_min FROM num_ranges) AND s.NTproBNP <= (SELECT ntprobnp_max FROM num_ranges)))
            AND (s.CREAT IS NULL OR (s.CREAT >= (SELECT creat_min FROM num_ranges) AND s.CREAT <= (SELECT creat_max FROM num_ranges)))
            AND (s.BUN IS NULL OR (s.BUN >= (SELECT bun_min FROM num_ranges) AND s.BUN <= (SELECT bun_max FROM num_ranges)))
            AND (s.POTASS IS NULL OR (s.POTASS >= (SELECT potass_min FROM num_ranges) AND s.POTASS <= (SELECT potass_max FROM num_ranges)))
            AND (s.CHOL IS NULL OR (s.CHOL >= (SELECT chol_min FROM num_ranges) AND s.CHOL <= (SELECT chol_max FROM num_ranges)))
            AND (s.HR IS NULL OR (s.HR >= (SELECT hr_min FROM num_ranges) AND s.HR <= (SELECT hr_max FROM num_ranges)))
            AND (s.SBP IS NULL OR (s.SBP >= (SELECT sbp_min FROM num_ranges) AND s.SBP <= (SELECT sbp_max FROM num_ranges)))
            AND (s.DBP IS NULL OR (s.DBP >= (SELECT dbp_min FROM num_ranges) AND s.DBP <= (SELECT dbp_max FROM num_ranges)))
            AND (s.RR IS NULL OR (s.RR >= (SELECT rr_min FROM num_ranges) AND s.RR <= (SELECT rr_max FROM num_ranges)))
        THEN 1 ELSE 0 END as passes_all_rules
    FROM synthetic_hashed s
),

-- ============================================================================
-- Plausibility Metrics: Count records passing all rules
-- ============================================================================

plausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity
    WHERE passes_all_rules = 1
),

plausible_unique AS (
    SELECT COUNT(DISTINCT row_hash) as count
    FROM synthetic_with_validity
    WHERE passes_all_rules = 1
),

implausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity
    WHERE passes_all_rules = 0
),

implausible_unique AS (
    SELECT COUNT(DISTINCT row_hash) as count
    FROM synthetic_with_validity
    WHERE passes_all_rules = 0
),

-- ============================================================================
-- 2x2 Matrix: Factual x Novel Quality Metrics
-- ============================================================================

-- Total Factual: Records in population (S ∩ P)
total_factual AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
),

-- Novel Plausible: Records passing validation and not in training (S ∩ V ∩ T̄)
novel_plausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.passes_all_rules = 1
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
),

-- ============================================================================
-- Cross-Tabulation: Category x Plausibility (Total counts only)
-- ============================================================================

-- DDR x Plausibility
ddr_plausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
      AND s.passes_all_rules = 1
),

ddr_implausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
      AND s.passes_all_rules = 0
),

-- Training Copy (Valid) x Plausibility
training_copy_valid_plausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
      AND s.row_hash IN (SELECT row_hash FROM population_hashes)
      AND s.passes_all_rules = 1
),

training_copy_valid_implausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
      AND s.row_hash IN (SELECT row_hash FROM population_hashes)
      AND s.passes_all_rules = 0
),

-- Training Copy (Propagation) x Plausibility
training_copy_propagation_plausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
      AND s.passes_all_rules = 1
),

training_copy_propagation_implausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.row_hash IN (SELECT row_hash FROM training_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
      AND s.passes_all_rules = 0
),

-- New Hallucination x Plausibility
new_hallucination_plausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
      AND s.passes_all_rules = 1
),

new_hallucination_implausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.row_hash NOT IN (SELECT row_hash FROM population_hashes)
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
      AND s.passes_all_rules = 0
)

-- ============================================================================
-- Final Summary: Combine all metrics
-- ============================================================================

SELECT
    -- Population dataset statistics
    ps.total as population_total_count,
    ps.unique_count as population_unique_count,

    -- Training dataset statistics
    ts.total as training_total_count,
    ts.unique_count as training_unique_count,

    -- Synthetic dataset statistics
    ss.total as synthetic_total_count,
    ss.unique_count as synthetic_unique_count,

    -- DDR metrics (desirable: in population but not in training)
    ddr_u.count as ddr_unique_count,
    ROUND((ddr_u.count::FLOAT / ss.unique_count * 100), 2) as ddr_unique_rate_pct,
    ddr_t.count as ddr_total_count,
    ROUND((ddr_t.count::FLOAT / ss.total * 100), 2) as ddr_total_rate_pct,

    -- Training Copy (Valid) metrics (copied from training AND in population)
    tcv_u.count as training_copy_valid_unique_count,
    ROUND((tcv_u.count::FLOAT / ss.unique_count * 100), 2) as training_copy_valid_unique_rate_pct,
    tcv_t.count as training_copy_valid_total_count,
    ROUND((tcv_t.count::FLOAT / ss.total * 100), 2) as training_copy_valid_total_rate_pct,

    -- Training Copy (Propagation) metrics (copied from training but NOT in population - hallucination propagation!)
    tcp_u.count as training_copy_propagation_unique_count,
    ROUND((tcp_u.count::FLOAT / ss.unique_count * 100), 2) as training_copy_propagation_unique_rate_pct,
    tcp_t.count as training_copy_propagation_total_count,
    ROUND((tcp_t.count::FLOAT / ss.total * 100), 2) as training_copy_propagation_total_rate_pct,

    -- New Hallucination metrics (NOT in population AND NOT in training - freshly fabricated)
    nh_u.count as new_hallucination_unique_count,
    ROUND((nh_u.count::FLOAT / ss.unique_count * 100), 2) as new_hallucination_unique_rate_pct,
    nh_t.count as new_hallucination_total_count,
    ROUND((nh_t.count::FLOAT / ss.total * 100), 2) as new_hallucination_total_rate_pct,

    -- Plausibility metrics (passing all validation rules)
    plaus_u.count as plausible_unique_count,
    ROUND((plaus_u.count::FLOAT / ss.unique_count * 100), 2) as plausible_unique_rate_pct,
    plaus_t.count as plausible_total_count,
    ROUND((plaus_t.count::FLOAT / ss.total * 100), 2) as plausible_total_rate_pct,

    -- Implausibility metrics (failing validation rules)
    impl_u.count as implausible_unique_count,
    ROUND((impl_u.count::FLOAT / ss.unique_count * 100), 2) as implausible_unique_rate_pct,
    impl_t.count as implausible_total_count,
    ROUND((impl_t.count::FLOAT / ss.total * 100), 2) as implausible_total_rate_pct,

    -- 2x2 Matrix: Factual x Novel Quality Metrics
    total_factual.count as total_factual_count,
    ROUND((total_factual.count::FLOAT / ss.total * 100), 2) as total_factual_rate_pct,
    ddr_t.count as novel_factual_count,  -- reuse DDR for Novel Factual (S ∩ P ∩ T̄)
    ROUND((ddr_t.count::FLOAT / ss.total * 100), 2) as novel_factual_rate_pct,
    plaus_t.count as total_plausible_count,  -- reuse for Total Plausible (S ∩ V)
    ROUND((plaus_t.count::FLOAT / ss.total * 100), 2) as total_plausible_rate_pct,
    novel_plausible.count as novel_plausible_count,
    ROUND((novel_plausible.count::FLOAT / ss.total * 100), 2) as novel_plausible_rate_pct,

    -- Cross-Tabulation: DDR x Plausibility (Total perspective only)
    ddr_plaus_t.count as ddr_plausible_total_count,
    ROUND((ddr_plaus_t.count::FLOAT / ss.total * 100), 2) as ddr_plausible_total_rate_pct,
    ddr_implaus_t.count as ddr_implausible_total_count,
    ROUND((ddr_implaus_t.count::FLOAT / ss.total * 100), 2) as ddr_implausible_total_rate_pct,

    -- Cross-Tabulation: Training Copy (Valid) x Plausibility
    tcv_plaus_t.count as training_copy_valid_plausible_total_count,
    ROUND((tcv_plaus_t.count::FLOAT / ss.total * 100), 2) as training_copy_valid_plausible_total_rate_pct,
    tcv_implaus_t.count as training_copy_valid_implausible_total_count,
    ROUND((tcv_implaus_t.count::FLOAT / ss.total * 100), 2) as training_copy_valid_implausible_total_rate_pct,

    -- Cross-Tabulation: Training Copy (Propagation) x Plausibility
    tcp_plaus_t.count as training_copy_propagation_plausible_total_count,
    ROUND((tcp_plaus_t.count::FLOAT / ss.total * 100), 2) as training_copy_propagation_plausible_total_rate_pct,
    tcp_implaus_t.count as training_copy_propagation_implausible_total_count,
    ROUND((tcp_implaus_t.count::FLOAT / ss.total * 100), 2) as training_copy_propagation_implausible_total_rate_pct,

    -- Cross-Tabulation: New Hallucination x Plausibility
    nh_plaus_t.count as new_hallucination_plausible_total_count,
    ROUND((nh_plaus_t.count::FLOAT / ss.total * 100), 2) as new_hallucination_plausible_total_rate_pct,
    nh_implaus_t.count as new_hallucination_implausible_total_count,
    ROUND((nh_implaus_t.count::FLOAT / ss.total * 100), 2) as new_hallucination_implausible_total_rate_pct

FROM population_stats ps
CROSS JOIN training_stats ts
CROSS JOIN synthetic_stats ss
CROSS JOIN ddr_unique ddr_u
CROSS JOIN ddr_total ddr_t
CROSS JOIN training_copy_valid_unique tcv_u
CROSS JOIN training_copy_valid_total tcv_t
CROSS JOIN training_copy_propagation_unique tcp_u
CROSS JOIN training_copy_propagation_total tcp_t
CROSS JOIN new_hallucination_unique nh_u
CROSS JOIN new_hallucination_total nh_t
CROSS JOIN plausible_unique plaus_u
CROSS JOIN plausible_total plaus_t
CROSS JOIN implausible_unique impl_u
CROSS JOIN implausible_total impl_t
CROSS JOIN total_factual
CROSS JOIN novel_plausible_total novel_plausible
CROSS JOIN ddr_plausible_total ddr_plaus_t
CROSS JOIN ddr_implausible_total ddr_implaus_t
CROSS JOIN training_copy_valid_plausible_total tcv_plaus_t
CROSS JOIN training_copy_valid_implausible_total tcv_implaus_t
CROSS JOIN training_copy_propagation_plausible_total tcp_plaus_t
CROSS JOIN training_copy_propagation_implausible_total tcp_implaus_t
CROSS JOIN new_hallucination_plausible_total nh_plaus_t
CROSS JOIN new_hallucination_implausible_total nh_implaus_t;
