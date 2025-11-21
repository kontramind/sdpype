-- ============================================================================
-- Unified Validation Queries for DDR and Plausibility Metrics
-- ============================================================================
--
-- DATASET: MIMIC-III ICU Stay Dataset
-- This validation file is DATASET-SPECIFIC and contains hardcoded column names
-- and validation rules for the MIMIC-III ICU stay dataset.
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
-- - Categorical membership: GENDER, ETHNICITY_GROUPED, ADMISSION_TYPE, IS_READMISSION_30D
-- - Numerical ranges: AGE, HR_FIRST, SYSBP_FIRST, DIASBP_FIRST, RESPRATE_FIRST,
--   NTPROBNP_FIRST, CREATININE_FIRST, BUN_FIRST, POTASSIUM_FIRST, TOTAL_CHOLESTEROL_FIRST
--   (within population min/max)
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
-- ============================================================================
-- Numerical Column Ranges from Population (for binning)
-- ============================================================================
num_ranges AS (
    SELECT
        MIN(AGE) as age_min, MAX(AGE) as age_max,
        MIN(HR_FIRST) as hr_min, MAX(HR_FIRST) as hr_max,
        MIN(SYSBP_FIRST) as sysbp_min, MAX(SYSBP_FIRST) as sysbp_max,
        MIN(DIASBP_FIRST) as diasbp_min, MAX(DIASBP_FIRST) as diasbp_max,
        MIN(RESPRATE_FIRST) as resprate_min, MAX(RESPRATE_FIRST) as resprate_max,
        MIN(NTPROBNP_FIRST) as ntprobnp_min, MAX(NTPROBNP_FIRST) as ntprobnp_max,
        MIN(CREATININE_FIRST) as creatinine_min, MAX(CREATININE_FIRST) as creatinine_max,
        MIN(BUN_FIRST) as bun_min, MAX(BUN_FIRST) as bun_max,
        MIN(POTASSIUM_FIRST) as potassium_min, MAX(POTASSIUM_FIRST) as potassium_max,
        MIN(TOTAL_CHOLESTEROL_FIRST) as cholesterol_min, MAX(TOTAL_CHOLESTEROL_FIRST) as cholesterol_max
    FROM population
),

-- ============================================================================
-- Binned Datasets for Factuality Checks (20 bins, NULL = bin 0)
-- ============================================================================
population_binned AS (
    SELECT
        -- Categorical columns (as-is)
        GENDER,
        ETHNICITY_GROUPED,
        ADMISSION_TYPE,
        IS_READMISSION_30D,
        -- Numerical columns (binned)
        COALESCE(WIDTH_BUCKET(AGE, r.age_min, r.age_max + 0.001, 20), 0) as AGE_BIN,
        COALESCE(WIDTH_BUCKET(HR_FIRST, r.hr_min, r.hr_max + 0.001, 20), 0) as HR_BIN,
        COALESCE(WIDTH_BUCKET(SYSBP_FIRST, r.sysbp_min, r.sysbp_max + 0.001, 20), 0) as SYSBP_BIN,
        COALESCE(WIDTH_BUCKET(DIASBP_FIRST, r.diasbp_min, r.diasbp_max + 0.001, 20), 0) as DIASBP_BIN,
        COALESCE(WIDTH_BUCKET(RESPRATE_FIRST, r.resprate_min, r.resprate_max + 0.001, 20), 0) as RESPRATE_BIN,
        COALESCE(WIDTH_BUCKET(NTPROBNP_FIRST, r.ntprobnp_min, r.ntprobnp_max + 0.001, 20), 0) as NTPROBNP_BIN,
        COALESCE(WIDTH_BUCKET(CREATININE_FIRST, r.creatinine_min, r.creatinine_max + 0.001, 20), 0) as CREATININE_BIN,
        COALESCE(WIDTH_BUCKET(BUN_FIRST, r.bun_min, r.bun_max + 0.001, 20), 0) as BUN_BIN,
        COALESCE(WIDTH_BUCKET(POTASSIUM_FIRST, r.potassium_min, r.potassium_max + 0.001, 20), 0) as POTASSIUM_BIN,
        COALESCE(WIDTH_BUCKET(TOTAL_CHOLESTEROL_FIRST, r.cholesterol_min, r.cholesterol_max + 0.001, 20), 0) as CHOLESTEROL_BIN
    FROM population, num_ranges r
),

training_binned AS (
    SELECT
        -- Categorical columns (as-is)
        GENDER,
        ETHNICITY_GROUPED,
        ADMISSION_TYPE,
        IS_READMISSION_30D,
        -- Numerical columns (binned)
        COALESCE(WIDTH_BUCKET(AGE, r.age_min, r.age_max + 0.001, 20), 0) as AGE_BIN,
        COALESCE(WIDTH_BUCKET(HR_FIRST, r.hr_min, r.hr_max + 0.001, 20), 0) as HR_BIN,
        COALESCE(WIDTH_BUCKET(SYSBP_FIRST, r.sysbp_min, r.sysbp_max + 0.001, 20), 0) as SYSBP_BIN,
        COALESCE(WIDTH_BUCKET(DIASBP_FIRST, r.diasbp_min, r.diasbp_max + 0.001, 20), 0) as DIASBP_BIN,
        COALESCE(WIDTH_BUCKET(RESPRATE_FIRST, r.resprate_min, r.resprate_max + 0.001, 20), 0) as RESPRATE_BIN,
        COALESCE(WIDTH_BUCKET(NTPROBNP_FIRST, r.ntprobnp_min, r.ntprobnp_max + 0.001, 20), 0) as NTPROBNP_BIN,
        COALESCE(WIDTH_BUCKET(CREATININE_FIRST, r.creatinine_min, r.creatinine_max + 0.001, 20), 0) as CREATININE_BIN,
        COALESCE(WIDTH_BUCKET(BUN_FIRST, r.bun_min, r.bun_max + 0.001, 20), 0) as BUN_BIN,
        COALESCE(WIDTH_BUCKET(POTASSIUM_FIRST, r.potassium_min, r.potassium_max + 0.001, 20), 0) as POTASSIUM_BIN,
        COALESCE(WIDTH_BUCKET(TOTAL_CHOLESTEROL_FIRST, r.cholesterol_min, r.cholesterol_max + 0.001, 20), 0) as CHOLESTEROL_BIN
    FROM training, num_ranges r
),

synthetic_binned AS (
    SELECT
        s.*,
        -- Categorical columns (as-is)
        s.GENDER as GENDER_CAT,
        s.ETHNICITY_GROUPED as ETHNICITY_CAT,
        s.ADMISSION_TYPE as ADMISSION_CAT,
        s.IS_READMISSION_30D as READMISSION_CAT,
        -- Numerical columns (binned)
        COALESCE(WIDTH_BUCKET(s.AGE, r.age_min, r.age_max + 0.001, 20), 0) as AGE_BIN,
        COALESCE(WIDTH_BUCKET(s.HR_FIRST, r.hr_min, r.hr_max + 0.001, 20), 0) as HR_BIN,
        COALESCE(WIDTH_BUCKET(s.SYSBP_FIRST, r.sysbp_min, r.sysbp_max + 0.001, 20), 0) as SYSBP_BIN,
        COALESCE(WIDTH_BUCKET(s.DIASBP_FIRST, r.diasbp_min, r.diasbp_max + 0.001, 20), 0) as DIASBP_BIN,
        COALESCE(WIDTH_BUCKET(s.RESPRATE_FIRST, r.resprate_min, r.resprate_max + 0.001, 20), 0) as RESPRATE_BIN,
        COALESCE(WIDTH_BUCKET(s.NTPROBNP_FIRST, r.ntprobnp_min, r.ntprobnp_max + 0.001, 20), 0) as NTPROBNP_BIN,
        COALESCE(WIDTH_BUCKET(s.CREATININE_FIRST, r.creatinine_min, r.creatinine_max + 0.001, 20), 0) as CREATININE_BIN,
        COALESCE(WIDTH_BUCKET(s.BUN_FIRST, r.bun_min, r.bun_max + 0.001, 20), 0) as BUN_BIN,
        COALESCE(WIDTH_BUCKET(s.POTASSIUM_FIRST, r.potassium_min, r.potassium_max + 0.001, 20), 0) as POTASSIUM_BIN,
        COALESCE(WIDTH_BUCKET(s.TOTAL_CHOLESTEROL_FIRST, r.cholesterol_min, r.cholesterol_max + 0.001, 20), 0) as CHOLESTEROL_BIN
    FROM synthetic s, num_ranges r
),

-- ============================================================================
-- Hash Datasets for Factuality Comparisons (using binned values)
-- ============================================================================
synthetic_hashed AS (
    SELECT
        *,
        hash(GENDER_CAT, ETHNICITY_CAT, ADMISSION_CAT, READMISSION_CAT,
             AGE_BIN, HR_BIN, SYSBP_BIN, DIASBP_BIN, RESPRATE_BIN,
             NTPROBNP_BIN, CREATININE_BIN, BUN_BIN, POTASSIUM_BIN, CHOLESTEROL_BIN) as row_hash
    FROM synthetic_binned
),

population_hashes AS (
    SELECT DISTINCT
        hash(GENDER, ETHNICITY_GROUPED, ADMISSION_TYPE, IS_READMISSION_30D,
             AGE_BIN, HR_BIN, SYSBP_BIN, DIASBP_BIN, RESPRATE_BIN,
             NTPROBNP_BIN, CREATININE_BIN, BUN_BIN, POTASSIUM_BIN, CHOLESTEROL_BIN) as row_hash
    FROM population_binned
),

training_hashes AS (
    SELECT DISTINCT
        hash(GENDER, ETHNICITY_GROUPED, ADMISSION_TYPE, IS_READMISSION_30D,
             AGE_BIN, HR_BIN, SYSBP_BIN, DIASBP_BIN, RESPRATE_BIN,
             NTPROBNP_BIN, CREATININE_BIN, BUN_BIN, POTASSIUM_BIN, CHOLESTEROL_BIN) as row_hash
    FROM training_binned
),

-- ============================================================================
-- Dataset Statistics
-- ============================================================================
population_stats AS (
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT row_hash) as unique_count
    FROM (SELECT hash(GENDER, ETHNICITY_GROUPED, ADMISSION_TYPE, IS_READMISSION_30D,
             AGE_BIN, HR_BIN, SYSBP_BIN, DIASBP_BIN, RESPRATE_BIN,
             NTPROBNP_BIN, CREATININE_BIN, BUN_BIN, POTASSIUM_BIN, CHOLESTEROL_BIN) as row_hash
          FROM population_binned)
),

training_stats AS (
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT row_hash) as unique_count
    FROM (SELECT hash(GENDER, ETHNICITY_GROUPED, ADMISSION_TYPE, IS_READMISSION_30D,
             AGE_BIN, HR_BIN, SYSBP_BIN, DIASBP_BIN, RESPRATE_BIN,
             NTPROBNP_BIN, CREATININE_BIN, BUN_BIN, POTASSIUM_BIN, CHOLESTEROL_BIN) as row_hash
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
-- Plausibility Rules: Extract valid values/ranges from population
-- ============================================================================

-- Valid categorical values from population
valid_genders AS (
    SELECT DISTINCT GENDER as value FROM population
),

valid_ethnicities AS (
    SELECT DISTINCT ETHNICITY_GROUPED as value FROM population
),

valid_admission_types AS (
    SELECT DISTINCT ADMISSION_TYPE as value FROM population
),

valid_readmission_flags AS (
    SELECT DISTINCT IS_READMISSION_30D as value FROM population
),

-- ============================================================================
-- Plausibility Validation: Check each synthetic record against all rules
-- ============================================================================

synthetic_with_validity AS (
    SELECT
        s.*,
        -- Check categorical rules
        CASE WHEN s.GENDER IN (SELECT value FROM valid_genders) THEN 1 ELSE 0 END as gender_valid,
        CASE WHEN s.ETHNICITY_GROUPED IN (SELECT value FROM valid_ethnicities) THEN 1 ELSE 0 END as ethnicity_valid,
        CASE WHEN s.ADMISSION_TYPE IN (SELECT value FROM valid_admission_types) THEN 1 ELSE 0 END as admission_type_valid,
        CASE WHEN s.IS_READMISSION_30D IN (SELECT value FROM valid_readmission_flags) THEN 1 ELSE 0 END as readmission_valid,
        -- Check numerical range rules (NULL is allowed, so check value OR NULL)
        CASE WHEN s.AGE IS NULL OR (s.AGE >= (SELECT age_min FROM num_ranges) AND s.AGE <= (SELECT age_max FROM num_ranges)) THEN 1 ELSE 0 END as age_valid,
        CASE WHEN s.HR_FIRST IS NULL OR (s.HR_FIRST >= (SELECT hr_min FROM num_ranges) AND s.HR_FIRST <= (SELECT hr_max FROM num_ranges)) THEN 1 ELSE 0 END as hr_valid,
        CASE WHEN s.SYSBP_FIRST IS NULL OR (s.SYSBP_FIRST >= (SELECT sysbp_min FROM num_ranges) AND s.SYSBP_FIRST <= (SELECT sysbp_max FROM num_ranges)) THEN 1 ELSE 0 END as sysbp_valid,
        CASE WHEN s.DIASBP_FIRST IS NULL OR (s.DIASBP_FIRST >= (SELECT diasbp_min FROM num_ranges) AND s.DIASBP_FIRST <= (SELECT diasbp_max FROM num_ranges)) THEN 1 ELSE 0 END as diasbp_valid,
        CASE WHEN s.RESPRATE_FIRST IS NULL OR (s.RESPRATE_FIRST >= (SELECT resprate_min FROM num_ranges) AND s.RESPRATE_FIRST <= (SELECT resprate_max FROM num_ranges)) THEN 1 ELSE 0 END as resprate_valid,
        CASE WHEN s.NTPROBNP_FIRST IS NULL OR (s.NTPROBNP_FIRST >= (SELECT ntprobnp_min FROM num_ranges) AND s.NTPROBNP_FIRST <= (SELECT ntprobnp_max FROM num_ranges)) THEN 1 ELSE 0 END as ntprobnp_valid,
        CASE WHEN s.CREATININE_FIRST IS NULL OR (s.CREATININE_FIRST >= (SELECT creatinine_min FROM num_ranges) AND s.CREATININE_FIRST <= (SELECT creatinine_max FROM num_ranges)) THEN 1 ELSE 0 END as creatinine_valid,
        CASE WHEN s.BUN_FIRST IS NULL OR (s.BUN_FIRST >= (SELECT bun_min FROM num_ranges) AND s.BUN_FIRST <= (SELECT bun_max FROM num_ranges)) THEN 1 ELSE 0 END as bun_valid,
        CASE WHEN s.POTASSIUM_FIRST IS NULL OR (s.POTASSIUM_FIRST >= (SELECT potassium_min FROM num_ranges) AND s.POTASSIUM_FIRST <= (SELECT potassium_max FROM num_ranges)) THEN 1 ELSE 0 END as potassium_valid,
        CASE WHEN s.TOTAL_CHOLESTEROL_FIRST IS NULL OR (s.TOTAL_CHOLESTEROL_FIRST >= (SELECT cholesterol_min FROM num_ranges) AND s.TOTAL_CHOLESTEROL_FIRST <= (SELECT cholesterol_max FROM num_ranges)) THEN 1 ELSE 0 END as cholesterol_valid,
        -- Check if ALL rules pass
        CASE WHEN
            s.GENDER IN (SELECT value FROM valid_genders)
            AND s.ETHNICITY_GROUPED IN (SELECT value FROM valid_ethnicities)
            AND s.ADMISSION_TYPE IN (SELECT value FROM valid_admission_types)
            AND s.IS_READMISSION_30D IN (SELECT value FROM valid_readmission_flags)
            AND (s.AGE IS NULL OR (s.AGE >= (SELECT age_min FROM num_ranges) AND s.AGE <= (SELECT age_max FROM num_ranges)))
            AND (s.HR_FIRST IS NULL OR (s.HR_FIRST >= (SELECT hr_min FROM num_ranges) AND s.HR_FIRST <= (SELECT hr_max FROM num_ranges)))
            AND (s.SYSBP_FIRST IS NULL OR (s.SYSBP_FIRST >= (SELECT sysbp_min FROM num_ranges) AND s.SYSBP_FIRST <= (SELECT sysbp_max FROM num_ranges)))
            AND (s.DIASBP_FIRST IS NULL OR (s.DIASBP_FIRST >= (SELECT diasbp_min FROM num_ranges) AND s.DIASBP_FIRST <= (SELECT diasbp_max FROM num_ranges)))
            AND (s.RESPRATE_FIRST IS NULL OR (s.RESPRATE_FIRST >= (SELECT resprate_min FROM num_ranges) AND s.RESPRATE_FIRST <= (SELECT resprate_max FROM num_ranges)))
            AND (s.NTPROBNP_FIRST IS NULL OR (s.NTPROBNP_FIRST >= (SELECT ntprobnp_min FROM num_ranges) AND s.NTPROBNP_FIRST <= (SELECT ntprobnp_max FROM num_ranges)))
            AND (s.CREATININE_FIRST IS NULL OR (s.CREATININE_FIRST >= (SELECT creatinine_min FROM num_ranges) AND s.CREATININE_FIRST <= (SELECT creatinine_max FROM num_ranges)))
            AND (s.BUN_FIRST IS NULL OR (s.BUN_FIRST >= (SELECT bun_min FROM num_ranges) AND s.BUN_FIRST <= (SELECT bun_max FROM num_ranges)))
            AND (s.POTASSIUM_FIRST IS NULL OR (s.POTASSIUM_FIRST >= (SELECT potassium_min FROM num_ranges) AND s.POTASSIUM_FIRST <= (SELECT potassium_max FROM num_ranges)))
            AND (s.TOTAL_CHOLESTEROL_FIRST IS NULL OR (s.TOTAL_CHOLESTEROL_FIRST >= (SELECT cholesterol_min FROM num_ranges) AND s.TOTAL_CHOLESTEROL_FIRST <= (SELECT cholesterol_max FROM num_ranges)))
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
