-- ==============================================================================
-- Advanced SQL Analytics: Process Optimization & Operational Insights
-- Tailored to demonstrate CTEs, Window Functions, and Cohort Aggregation
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. WINDOW FUNCTION: Ranking Optimal Operating Runs per Regime
-- Identifies the top 3 most energy-efficient runs that achieved on-spec purity
-- ------------------------------------------------------------------------------
WITH RankedRuns AS (
    SELECT 
        run_id,
        regime_id,
        feed_flow_kmol_hr,
        reflux_ratio,
        reboiler_duty_mw,
        distillate_c3_purity,
        specific_energy_gj_ton,
        hourly_operating_cost_usd,
        ROW_NUMBER() OVER (
            PARTITION BY regime_id 
            ORDER BY specific_energy_gj_ton ASC
        ) as efficiency_rank,
        AVG(specific_energy_gj_ton) OVER (
            PARTITION BY regime_id
        ) as regime_avg_energy
    FROM simulation_runs
    WHERE is_on_spec = 1
)
SELECT 
    regime_id,
    efficiency_rank,
    run_id,
    reflux_ratio,
    reboiler_duty_mw,
    distillate_c3_purity,
    specific_energy_gj_ton,
    ROUND(regime_avg_energy - specific_energy_gj_ton, 3) as energy_savings_vs_avg
FROM RankedRuns
WHERE efficiency_rank <= 3
ORDER BY regime_id, efficiency_rank;

-- ------------------------------------------------------------------------------
-- 2. CTE + AGGREGATION: Pareto Frontier for Energy vs Purity
-- Groups runs into purity brackets and identifies minimum reboiler requirements
-- ------------------------------------------------------------------------------
WITH PurityBuckets AS (
    SELECT 
        run_id,
        reboiler_duty_mw,
        reflux_ratio,
        specific_energy_gj_ton,
        hourly_operating_cost_usd,
        CASE 
            WHEN distillate_c3_purity >= 0.998 THEN 'Tier 1: High Purity (>=99.8%)'
            WHEN distillate_c3_purity >= 0.995 THEN 'Tier 2: Standard On-Spec (99.5%-99.8%)'
            WHEN distillate_c3_purity >= 0.990 THEN 'Tier 3: Marginal (99.0%-99.5%)'
            ELSE 'Tier 4: Off-Spec (<99.0%)'
        END as purity_tier
    FROM simulation_runs
)
SELECT 
    purity_tier,
    COUNT(*) as total_runs,
    ROUND(MIN(reboiler_duty_mw), 2) as min_reboiler_duty_mw,
    ROUND(AVG(reboiler_duty_mw), 2) as avg_reboiler_duty_mw,
    ROUND(AVG(specific_energy_gj_ton), 3) as avg_specific_energy_gj_ton,
    ROUND(AVG(hourly_operating_cost_usd), 2) as avg_operating_cost_usd,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY reboiler_duty_mw), 2) as median_reboiler_duty
FROM PurityBuckets
GROUP BY purity_tier
ORDER BY avg_specific_energy_gj_ton DESC;

-- ------------------------------------------------------------------------------
-- 3. MOVING WINDOW & DRIFT: Detecting Process Instability in Historian Data
-- Rolling 7-run moving average and standard deviation of column pressure
-- ------------------------------------------------------------------------------
SELECT 
    run_id,
    timestamp,
    column_pressure_bar,
    AVG(column_pressure_bar) OVER (
        ORDER BY timestamp 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as rolling_avg_pressure,
    reboiler_duty_mw,
    distillate_c3_purity,
    CASE 
        WHEN ABS(column_pressure_bar - AVG(column_pressure_bar) OVER (
            ORDER BY timestamp ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )) > 0.5 THEN 'DRIFT_ALERT'
        ELSE 'NORMAL'
    END as stability_status
FROM simulation_runs
ORDER BY timestamp DESC
LIMIT 50;

-- ------------------------------------------------------------------------------
-- 4. BUSINESS IMPACT QUERY: Annualized Energy Savings Opportunity ("Now What")
-- Calculates potential economic uplift if plant operates at 10th percentile energy
-- ------------------------------------------------------------------------------
WITH BaselineVsOptimal AS (
    SELECT 
        AVG(hourly_operating_cost_usd) as baseline_hourly_cost,
        (
            SELECT AVG(hourly_operating_cost_usd) 
            FROM (
                SELECT hourly_operating_cost_usd 
                FROM simulation_runs 
                WHERE is_on_spec = 1 
                ORDER BY specific_energy_gj_ton ASC 
                LIMIT 50
            )
        ) as best_in_class_hourly_cost
    FROM simulation_runs
    WHERE is_on_spec = 1
)
SELECT 
    ROUND(baseline_hourly_cost, 2) as baseline_hourly_cost_usd,
    ROUND(best_in_class_hourly_cost, 2) as optimized_hourly_cost_usd,
    ROUND(baseline_hourly_cost - best_in_class_hourly_cost, 2) as hourly_savings_usd,
    ROUND((baseline_hourly_cost - best_in_class_hourly_cost) * 8400, 2) as annual_savings_usd,
    ROUND(((baseline_hourly_cost - best_in_class_hourly_cost) / baseline_hourly_cost) * 100, 2) as energy_cost_reduction_pct
FROM BaselineVsOptimal;
