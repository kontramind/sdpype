-- DDR Summary: Compute all metrics in a single query
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
