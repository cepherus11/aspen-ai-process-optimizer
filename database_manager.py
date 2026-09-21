"""
Database Management & Ingestion Layer
Manages relational schema initialization, simulation run ingestion,
and analytical SQL execution using SQLite / PostgreSQL.
"""

import sqlite3
import pandas as pd
import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class ProcessDatabaseManager:
    """
    Manages process simulation data warehouse and execution of analytical queries.
    """

    def __init__(self, db_path: str = "data/process_historian.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Returns SQLite database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self, schema_sql_path: Optional[str] = None):
        """
        Executes schema DDL to create tables and default master data.
        """
        if schema_sql_path is None:
            # Default to sql/schema.sql
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            schema_sql_path = os.path.join(base_dir, "sql", "schema.sql")

        with open(schema_sql_path, "r") as f:
            ddl_script = f.read()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(ddl_script)

            # Insert default unit and regimes if not present
            cursor.execute(
                """
                INSERT OR IGNORE INTO process_units (unit_id, unit_name, unit_type, design_capacity_kmol_hr, installation_year)
                VALUES ('C-301_DEPROPANIZER', 'Main Hydrocarbon Fractionation Train', 'Distillation Column', 1500.0, 2021);
            """
            )

            cursor.executemany(
                """
                INSERT OR IGNORE INTO operating_regimes (regime_id, regime_name, description, target_purity_min, max_energy_budget_mw)
                VALUES (?, ?, ?, ?, ?);
            """,
                [
                    (
                        "REGIME_STANDARD",
                        "Nominal Base Load",
                        "Standard operating window under design feed rate and composition",
                        0.995,
                        22.0,
                    ),
                    (
                        "REGIME_HIGH_THROUGHPUT",
                        "Maximum Capacity",
                        "Operation exceeding 1200 kmol/hr feed rate to maximize throughput",
                        0.995,
                        28.0,
                    ),
                    (
                        "REGIME_LEAN_FEED",
                        "Low C3 Feedstock",
                        "Feed containing < 70% C3 requiring tighter fractionation control",
                        0.995,
                        20.0,
                    ),
                ],
            )
            conn.commit()
            logger.info("Database schema and master records initialized.")

    def ingest_simulation_runs(self, df: pd.DataFrame) -> int:
        """
        Ingests simulation runs into the database.
        """
        with self.get_connection() as conn:
            df.to_sql("simulation_runs", conn, if_exists="append", index=False)
            logger.info(f"Ingested {len(df)} simulation runs into database.")
            return len(df)

    def query_to_dataframe(self, sql_query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Executes a SQL query and returns results as a Pandas DataFrame.
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(sql_query, conn, params=params)

    def log_optimization(
        self,
        opt_id: str,
        run_id: str,
        orig_reboiler: float,
        opt_reboiler: float,
        savings_pct: float,
        cost_savings: float,
        status: str,
    ):
        """
        Logs an optimization event into the audit table.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO optimization_audit 
                (opt_id, run_id, original_reboiler_mw, optimized_reboiler_mw, energy_savings_pct, cost_savings_hourly_usd, solver_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (opt_id, run_id, orig_reboiler, opt_reboiler, savings_pct, cost_savings, status),
            )
            conn.commit()


if __name__ == "__main__":
    db = ProcessDatabaseManager(db_path="/tmp/test_historian.db")
    db.initialize_schema()
    print("Database manager tested successfully.")
