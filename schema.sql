-- ==============================================================================
-- Schema: Chemical Process Historian & Simulation Data Warehouse
-- ==============================================================================

-- 1. Process Units Table
CREATE TABLE IF NOT EXISTS process_units (
    unit_id VARCHAR(20) PRIMARY KEY,
    unit_name VARCHAR(100) NOT NULL,
    unit_type VARCHAR(50) NOT NULL,
    design_capacity_kmol_hr REAL NOT NULL,
    installation_year INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Operating Regimes Definition
CREATE TABLE IF NOT EXISTS operating_regimes (
    regime_id VARCHAR(20) PRIMARY KEY,
    regime_name VARCHAR(100) NOT NULL,
    description TEXT,
    target_purity_min REAL NOT NULL,
    max_energy_budget_mw REAL NOT NULL
);

-- 3. Simulation & Plant Historian Runs
CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    unit_id VARCHAR(20) NOT NULL,
    regime_id VARCHAR(20) NOT NULL,
    
    -- Manipulated & Disturbance Variables (Inputs)
    feed_flow_kmol_hr REAL NOT NULL,
    feed_temp_c REAL NOT NULL,
    feed_c3_fraction REAL NOT NULL,
    reflux_ratio REAL NOT NULL,
    reboiler_duty_mw REAL NOT NULL,
    column_pressure_bar REAL NOT NULL,
    
    -- Measured & Simulated Outputs (Targets)
    distillate_c3_purity REAL NOT NULL,
    bottoms_c3_loss REAL NOT NULL,
    condenser_duty_mw REAL NOT NULL,
    specific_energy_gj_ton REAL NOT NULL,
    hourly_operating_cost_usd REAL NOT NULL,
    
    -- Status & Quality Flags
    is_on_spec BOOLEAN NOT NULL,
    convergence_time_sec REAL NOT NULL,
    source_engine VARCHAR(30) DEFAULT 'Aspen_Plus_V14',
    
    FOREIGN KEY (unit_id) REFERENCES process_units (unit_id),
    FOREIGN KEY (regime_id) REFERENCES operating_regimes (regime_id)
);

-- 4. Optimization Audit Log
CREATE TABLE IF NOT EXISTS optimization_audit (
    opt_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_id VARCHAR(50),
    original_reboiler_mw REAL NOT NULL,
    optimized_reboiler_mw REAL NOT NULL,
    energy_savings_pct REAL NOT NULL,
    cost_savings_hourly_usd REAL NOT NULL,
    solver_status VARCHAR(20) NOT NULL,
    FOREIGN KEY (run_id) REFERENCES simulation_runs (run_id)
);

-- Indexes for high-performance analytical queries
CREATE INDEX IF NOT EXISTS idx_sim_regime ON simulation_runs(regime_id);
CREATE INDEX IF NOT EXISTS idx_sim_purity ON simulation_runs(distillate_c3_purity);
CREATE INDEX IF NOT EXISTS idx_sim_energy ON simulation_runs(specific_energy_gj_ton);
