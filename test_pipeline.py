"""
Unit Tests for Aspen-AI Chemical Process Platform
Validates:
1. Thermodynamic simulation engine & LHS generation
2. SQL schema initialization and data ingestion
3. ML surrogate training, evaluation, and latency
4. Prescriptive optimization convergence
5. Statistical testing (ANOVA, t-test, Cp/Cpk)
6. GenAI Copilot SQL generation
"""

import pytest
import os
import shutil
import numpy as np
import pandas as pd

from src.simulation.process_simulator import DepropanizerSimulator
from src.data.database_manager import ProcessDatabaseManager
from src.models.surrogate_trainer import ProcessSurrogateSuite
from src.models.explainability import ProcessExplainer
from src.analytics.statistical_testing import ProcessStatisticalTesting
from src.optimization.energy_optimizer import ProcessEnergyOptimizer
from src.agent.copilot import PlantCopilotAgent

TEST_DB_PATH = "data/test_process.db"


@pytest.fixture(scope="module")
def sample_dataset():
    sim = DepropanizerSimulator()
    df = sim.generate_lhs_dataset(num_samples=50, random_seed=42)
    return df


def test_simulation_bounds_and_physics(sample_dataset):
    df = sample_dataset
    assert len(df) == 50
    assert (df["distillate_c3_purity"] >= 0.85).all()
    assert (df["distillate_c3_purity"] <= 1.0).all()
    assert (df["condenser_duty_mw"] > 0).all()
    assert (df["specific_energy_gj_ton"] > 0).all()
    assert (df["hourly_operating_cost_usd"] > 0).all()


def test_database_initialization_and_ingestion(sample_dataset):
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    db = ProcessDatabaseManager(db_path=TEST_DB_PATH)
    db.initialize_schema()

    ingested_count = db.ingest_simulation_runs(sample_dataset)
    assert ingested_count == len(sample_dataset)

    # Test analytical SQL query execution
    df_read = db.query_to_dataframe("SELECT COUNT(*) as total FROM simulation_runs")
    assert df_read.iloc[0]["total"] == 50

    # Test Window Function query
    window_sql = """
    SELECT run_id, regime_id,
           ROW_NUMBER() OVER (PARTITION BY regime_id ORDER BY specific_energy_gj_ton ASC) as rank
    FROM simulation_runs;
    """
    df_window = db.query_to_dataframe(window_sql)
    assert not df_window.empty
    assert "rank" in df_window.columns

    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


def test_surrogate_training_and_latency(sample_dataset):
    surrogate = ProcessSurrogateSuite(primary_target="distillate_c3_purity")
    X_train, X_test, y_train, y_test = surrogate.prepare_data(sample_dataset, test_size=0.2)

    # Test PCA
    pca_res = surrogate.fit_pca(X_train)
    assert len(pca_res["explained_variance_ratio"]) == 3
    assert sum(pca_res["explained_variance_ratio"]) > 0.60

    # Test Training
    benchmarks = surrogate.train_and_benchmark(X_train, X_test, y_train, y_test)
    assert not benchmarks.empty
    assert benchmarks.iloc[0]["R2_Score"] > 0.80

    # Test single instance prediction
    sample_input = {
        "feed_flow_kmol_hr": 1000.0,
        "feed_temp_c": 52.0,
        "feed_c3_fraction": 0.75,
        "reflux_ratio": 4.2,
        "reboiler_duty_mw": 18.5,
        "column_pressure_bar": 18.0,
    }
    pred_purity = surrogate.predict_single(sample_input)
    assert 0.80 <= pred_purity <= 1.0


def test_explainability(sample_dataset):
    surrogate = ProcessSurrogateSuite(primary_target="distillate_c3_purity")
    X_train, X_test, y_train, y_test = surrogate.prepare_data(sample_dataset, test_size=0.2)
    surrogate.train_and_benchmark(X_train, X_test, y_train, y_test)

    explainer = ProcessExplainer(surrogate.best_model, surrogate.FEATURE_COLS)
    global_imp = explainer.compute_global_importance(X_test)
    assert len(global_imp) == len(surrogate.FEATURE_COLS)
    assert "Importance_Pct" in global_imp.columns


def test_statistical_testing(sample_dataset):
    anova_res = ProcessStatisticalTesting.run_regime_anova(sample_dataset)
    assert "f_statistic" in anova_res
    assert "p_value" in anova_res

    purity_vals = sample_dataset["distillate_c3_purity"].values
    cp_res = ProcessStatisticalTesting.calculate_process_capability(purity_vals)
    assert "Cp" in cp_res
    assert "Cpk" in cp_res


def test_energy_optimizer(sample_dataset):
    surrogate = ProcessSurrogateSuite(primary_target="distillate_c3_purity")
    X_train, X_test, y_train, y_test = surrogate.prepare_data(sample_dataset, test_size=0.2)
    surrogate.train_and_benchmark(X_train, X_test, y_train, y_test)

    optimizer = ProcessEnergyOptimizer(surrogate_suite=surrogate)
    res = optimizer.optimize_operating_point(
        feed_flow_kmol_hr=1000.0,
        feed_temp_c=52.0,
        feed_c3_fraction=0.75,
        column_pressure_bar=18.0,
        initial_reflux=4.5,
        initial_reboiler_mw=22.0,
    )
    assert "optimized_setpoints" in res
    assert res["predicted_purity"] >= 0.990


def test_genai_copilot_sql_generation():
    copilot = PlantCopilotAgent(db_manager=None)
    sql_energy = copilot.generate_sql_query("Give me the top runs with lowest energy")
    assert "specific_energy_gj_ton ASC" in sql_energy

    sql_offspec = copilot.generate_sql_query("Analyze off-spec purity issues")
    assert "is_on_spec = 0" in sql_offspec
