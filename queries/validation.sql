-- ============================================================================
-- Unified Validation Queries for DDR and Plausibility Metrics
-- ============================================================================
--
-- DATASET: Canada COVID-19 Case Details
-- This validation file is DATASET-SPECIFIC and contains hardcoded column names
-- and validation rules for the Canada COVID-19 dataset.
--
-- For other datasets, create a separate validation SQL file with appropriate
-- column names and validation rules (e.g., queries/validation_adult.sql).
--
-- This SQL file contains queries to compute:
-- 1. DDR (Desirable Diverse Records) metrics - records that exist in population
--    but not in training data
-- 2. Plausibility metrics - records that pass all validation rules
--
-- Both metrics are computed with dual perspectives:
-- - Unique: Count of distinct records (by hash)
-- - Total: Count of all records including duplicates
--
-- Template placeholders:
-- - {{HASH_COLS}}: Replaced with comma-separated quoted column names at runtime
--
-- Validation Rules for COVID-19 Dataset:
-- - Categorical membership: Age Group, Gender, Exposure Type, Case Status,
--   Province Abbreviation, Region ID
-- - Date ranges: Date Reported (within population bounds)
-- - Combination constraints: Province Abbreviation + Region ID must be valid pair
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
WITH
-- Hash all datasets for efficient set operations
synthetic_hashed AS (
    SELECT *, hash({{HASH_COLS}}) as row_hash FROM synthetic
),

population_hashes AS (
    SELECT DISTINCT hash({{HASH_COLS}}) as row_hash FROM population
),

training_hashes AS (
    SELECT DISTINCT hash({{HASH_COLS}}) as row_hash FROM training
),

-- ============================================================================
-- Dataset Statistics
-- ============================================================================
population_stats AS (
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT hash({{HASH_COLS}})) as unique_count
    FROM population
),

training_stats AS (
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT hash({{HASH_COLS}})) as unique_count
    FROM training
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
-- Plausibility Rules: Extract valid values/ranges from population
-- ============================================================================

-- Valid categorical values from population
valid_age_groups AS (
    SELECT DISTINCT "Age Group" as value FROM population
),

valid_genders AS (
    SELECT DISTINCT "Gender" as value FROM population
),

valid_exposure_types AS (
    SELECT DISTINCT "Exposure Type" as value FROM population
),

valid_case_statuses AS (
    SELECT DISTINCT "Case Status" as value FROM population
),

valid_province_abbrevs AS (
    SELECT DISTINCT "Province Abbreviation" as value FROM population
),

valid_region_ids AS (
    SELECT DISTINCT "Region ID" as value FROM population
),

-- Valid date range from population
date_range AS (
    SELECT
        MIN("Date Reported") as min_date,
        MAX("Date Reported") as max_date
    FROM population
),

-- Valid Province Abbreviation + Region ID combinations from population
valid_province_region_combos AS (
    SELECT DISTINCT
        "Province Abbreviation",
        "Region ID"
    FROM population
),

-- ============================================================================
-- Plausibility Validation: Check each synthetic record against all rules
-- ============================================================================

synthetic_with_validity AS (
    SELECT
        s.*,
        s.row_hash,
        -- Check all categorical rules
        CASE WHEN s."Age Group" IN (SELECT value FROM valid_age_groups) THEN 1 ELSE 0 END as age_group_valid,
        CASE WHEN s."Gender" IN (SELECT value FROM valid_genders) THEN 1 ELSE 0 END as gender_valid,
        CASE WHEN s."Exposure Type" IN (SELECT value FROM valid_exposure_types) THEN 1 ELSE 0 END as exposure_type_valid,
        CASE WHEN s."Case Status" IN (SELECT value FROM valid_case_statuses) THEN 1 ELSE 0 END as case_status_valid,
        CASE WHEN s."Province Abbreviation" IN (SELECT value FROM valid_province_abbrevs) THEN 1 ELSE 0 END as province_abbrev_valid,
        CASE WHEN s."Region ID" IN (SELECT value FROM valid_region_ids) THEN 1 ELSE 0 END as region_id_valid,
        -- Check date range rule
        CASE WHEN s."Date Reported" BETWEEN (SELECT min_date FROM date_range)
                                        AND (SELECT max_date FROM date_range)
             THEN 1 ELSE 0 END as date_reported_valid,
        -- Check combination rule
        CASE WHEN EXISTS (
            SELECT 1 FROM valid_province_region_combos v
            WHERE v."Province Abbreviation" = s."Province Abbreviation"
              AND v."Region ID" = s."Region ID"
        ) THEN 1 ELSE 0 END as province_region_combo_valid,
        -- Check if ALL rules pass
        CASE WHEN
            s."Age Group" IN (SELECT value FROM valid_age_groups)
            AND s."Gender" IN (SELECT value FROM valid_genders)
            AND s."Exposure Type" IN (SELECT value FROM valid_exposure_types)
            AND s."Case Status" IN (SELECT value FROM valid_case_statuses)
            AND s."Province Abbreviation" IN (SELECT value FROM valid_province_abbrevs)
            AND s."Region ID" IN (SELECT value FROM valid_region_ids)
            AND s."Date Reported" BETWEEN (SELECT min_date FROM date_range)
                                      AND (SELECT max_date FROM date_range)
            AND EXISTS (
                SELECT 1 FROM valid_province_region_combos v
                WHERE v."Province Abbreviation" = s."Province Abbreviation"
                  AND v."Region ID" = s."Region ID"
            )
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
-- 2×2 Matrix: Factual × Novel Quality Metrics
-- ============================================================================

-- Total Factual: Records in population (S ∩ P)
total_factual AS (
    SELECT COUNT(*) as count
    FROM synthetic_hashed s
    WHERE s.row_hash IN (SELECT row_hash FROM population_hashes)
),

-- Novel Factual: Records in population but not in training (S ∩ P ∩ T̄) - same as DDR
-- (already computed as ddr_total above)

-- Total Plausible: Records passing validation (S ∩ V)
-- (already computed as plausible_total above)

-- Novel Plausible: Records passing validation and not in training (S ∩ V ∩ T̄)
novel_plausible_total AS (
    SELECT COUNT(*) as count
    FROM synthetic_with_validity s
    WHERE s.passes_all_rules = 1
      AND s.row_hash NOT IN (SELECT row_hash FROM training_hashes)
),

-- ============================================================================
-- Cross-Tabulation: Category × Plausibility (Total counts only)
-- ============================================================================

-- DDR × Plausibility
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

-- Training Copy (Valid) × Plausibility
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

-- Training Copy (Propagation) × Plausibility
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

-- New Hallucination × Plausibility
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

    -- 2×2 Matrix: Factual × Novel Quality Metrics
    total_factual.count as total_factual_count,
    ROUND((total_factual.count::FLOAT / ss.total * 100), 2) as total_factual_rate_pct,
    ddr_t.count as novel_factual_count,  -- reuse DDR for Novel Factual (S ∩ P ∩ T̄)
    ROUND((ddr_t.count::FLOAT / ss.total * 100), 2) as novel_factual_rate_pct,
    plaus_t.count as total_plausible_count,  -- reuse for Total Plausible (S ∩ V)
    ROUND((plaus_t.count::FLOAT / ss.total * 100), 2) as total_plausible_rate_pct,
    novel_plausible.count as novel_plausible_count,
    ROUND((novel_plausible.count::FLOAT / ss.total * 100), 2) as novel_plausible_rate_pct,

    -- Cross-Tabulation: DDR × Plausibility (Total perspective only)
    ddr_plaus_t.count as ddr_plausible_total_count,
    ROUND((ddr_plaus_t.count::FLOAT / ss.total * 100), 2) as ddr_plausible_total_rate_pct,
    ddr_implaus_t.count as ddr_implausible_total_count,
    ROUND((ddr_implaus_t.count::FLOAT / ss.total * 100), 2) as ddr_implausible_total_rate_pct,

    -- Cross-Tabulation: Training Copy (Valid) × Plausibility
    tcv_plaus_t.count as training_copy_valid_plausible_total_count,
    ROUND((tcv_plaus_t.count::FLOAT / ss.total * 100), 2) as training_copy_valid_plausible_total_rate_pct,
    tcv_implaus_t.count as training_copy_valid_implausible_total_count,
    ROUND((tcv_implaus_t.count::FLOAT / ss.total * 100), 2) as training_copy_valid_implausible_total_rate_pct,

    -- Cross-Tabulation: Training Copy (Propagation) × Plausibility
    tcp_plaus_t.count as training_copy_propagation_plausible_total_count,
    ROUND((tcp_plaus_t.count::FLOAT / ss.total * 100), 2) as training_copy_propagation_plausible_total_rate_pct,
    tcp_implaus_t.count as training_copy_propagation_implausible_total_count,
    ROUND((tcp_implaus_t.count::FLOAT / ss.total * 100), 2) as training_copy_propagation_implausible_total_rate_pct,

    -- Cross-Tabulation: New Hallucination × Plausibility
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
