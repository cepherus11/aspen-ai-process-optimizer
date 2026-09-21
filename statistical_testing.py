"""
Statistical Process Control & Hypothesis Testing Suite
Implements ANOVA, Two-Sample t-tests, A/B Testing, and Process Capability (Cp, Cpk).
Fulfills Tredence JD's statistical rigor requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ProcessStatisticalTesting:
    """
    Executes formal statistical hypothesis testing and process capability calculations
    on plant historian and simulation datasets.
    """

    @staticmethod
    def run_regime_anova(df: pd.DataFrame, metric: str = "specific_energy_gj_ton") -> Dict[str, Any]:
        """
        One-way ANOVA (Analysis of Variance) testing whether mean energy consumption
        differs significantly across operating regimes.
        H0: mu_standard = mu_high_throughput = mu_lean_feed
        H1: At least one regime has a significantly different mean
        """
        regimes = df["regime_id"].unique()
        groups = [df[df["regime_id"] == r][metric].dropna().values for r in regimes]

        f_stat, p_value = stats.f_oneway(*groups)

        regime_stats = {}
        for r in regimes:
            vals = df[df["regime_id"] == r][metric].dropna()
            regime_stats[r] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "count": int(len(vals)),
            }

        return {
            "test_name": "One-Way ANOVA",
            "metric": metric,
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "is_statistically_significant": bool(p_value < 0.05),
            "regime_summary": regime_stats,
            "interpretation": (
                "Statistically significant difference detected between operating regimes (p < 0.05)."
                if p_value < 0.05
                else "No statistically significant difference between regimes."
            ),
        }

    @staticmethod
    def run_two_sample_ttest(
        group_a: np.ndarray,
        group_b: np.ndarray,
        label_a: str = "Baseline",
        label_b: str = "Optimized",
    ) -> Dict[str, Any]:
        """
        Welch's Two-Sample t-test comparing two operating policies (e.g. A/B testing).
        """
        t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)

        mean_a = float(np.mean(group_a))
        mean_b = float(np.mean(group_b))
        diff = mean_a - mean_b
        pct_improvement = (diff / mean_a) * 100.0 if mean_a != 0 else 0.0

        return {
            "test_name": "Welch's Two-Sample t-test",
            "group_a": {"name": label_a, "mean": mean_a, "std": float(np.std(group_a))},
            "group_b": {"name": label_b, "mean": mean_b, "std": float(np.std(group_b))},
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "is_significant": bool(p_val < 0.01),
            "delta_mean": float(diff),
            "improvement_pct": float(pct_improvement),
        }

    @staticmethod
    def calculate_process_capability(
        purity_values: np.ndarray, lsl: float = 0.995, usl: float = 1.000
    ) -> Dict[str, float]:
        """
        Calculates Process Capability indices (Cp, Cpk) for product purity.
        Cp = (USL - LSL) / (6 * sigma)
        Cpk = min((USL - mu) / (3 * sigma), (mu - LSL) / (3 * sigma))
        """
        mu = float(np.mean(purity_values))
        sigma = float(np.std(purity_values, ddof=1))

        if sigma == 0:
            return {"Cp": 999.0, "Cpk": 999.0, "mean": mu, "sigma": 0.0}

        cp = (usl - lsl) / (6 * sigma)
        cpu = (usl - mu) / (3 * sigma)
        cpl = (mu - lsl) / (3 * sigma)
        cpk = min(cpu, cpl)

        return {
            "mean": mu,
            "std_dev": sigma,
            "Cp": float(cp),
            "Cpk": float(cpk),
            "is_capable_process": bool(cpk >= 1.33),  # Standard 4-sigma industrial quality threshold
        }
