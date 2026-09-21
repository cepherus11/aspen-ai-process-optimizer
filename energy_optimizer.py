"""
Prescriptive Optimization Engine: Energy & Operating Cost Minimization
Uses Scipy SLSQP constrained optimization coupled with the ML surrogate model
to find minimum-energy operational setpoints satisfying purity constraints.
"""

import numpy as np
from typing import Dict, Any, Tuple
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class ProcessEnergyOptimizer:
    """
    Solves non-linear constrained optimization to minimize reboiler duty (steam cost)
    subject to meeting the target distillate purity constraint (>= 99.5%).
    """

    def __init__(self, surrogate_suite: Any, target_purity_min: float = 0.995):
        self.surrogate = surrogate_suite
        self.target_purity_min = target_purity_min
        self.steam_cost_per_mwh = 42.50
        self.annual_hours = 8400

    def optimize_operating_point(
        self,
        feed_flow_kmol_hr: float,
        feed_temp_c: float,
        feed_c3_fraction: float,
        column_pressure_bar: float,
        initial_reflux: float = 4.5,
        initial_reboiler_mw: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Decision Variables: [reflux_ratio, reboiler_duty_mw]
        Objective: Minimize reboiler_duty_mw
        Constraint: surrogate_predicted_purity >= target_purity_min
        """
        # Feature names in exact order
        feature_cols = self.surrogate.FEATURE_COLS

        # Objective Function: Minimize reboiler duty + penalty for purity violation
        def objective(vars_opt):
            reflux, reboiler = vars_opt
            input_dict = {
                "feed_flow_kmol_hr": feed_flow_kmol_hr,
                "feed_temp_c": feed_temp_c,
                "feed_c3_fraction": feed_c3_fraction,
                "reflux_ratio": reflux,
                "reboiler_duty_mw": reboiler,
                "column_pressure_bar": column_pressure_bar,
            }
            pred_purity = self.surrogate.predict_single(input_dict)

            # Penalty for purity violation
            penalty = 0.0
            if pred_purity < self.target_purity_min:
                penalty = 10000.0 * (self.target_purity_min - pred_purity) ** 2

            return reboiler + penalty

        # Inequality Constraint: purity - target_purity_min >= 0
        def purity_constraint(vars_opt):
            reflux, reboiler = vars_opt
            input_dict = {
                "feed_flow_kmol_hr": feed_flow_kmol_hr,
                "feed_temp_c": feed_temp_c,
                "feed_c3_fraction": feed_c3_fraction,
                "reflux_ratio": reflux,
                "reboiler_duty_mw": reboiler,
                "column_pressure_bar": column_pressure_bar,
            }
            return self.surrogate.predict_single(input_dict) - self.target_purity_min

        # Bounds: Reflux [2.5, 6.5], Reboiler [12.0, 28.0]
        bounds = [(2.5, 6.5), (12.0, 28.0)]
        x0 = [initial_reflux, initial_reboiler_mw]

        constraints = [{"type": "ineq", "fun": purity_constraint}]

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-4, "maxiter": 100},
        )

        opt_reflux, opt_reboiler = res.x
        opt_input = {
            "feed_flow_kmol_hr": feed_flow_kmol_hr,
            "feed_temp_c": feed_temp_c,
            "feed_c3_fraction": feed_c3_fraction,
            "reflux_ratio": float(opt_reflux),
            "reboiler_duty_mw": float(opt_reboiler),
            "column_pressure_bar": column_pressure_bar,
        }
        optimized_purity = self.surrogate.predict_single(opt_input)

        # Economic calculation
        baseline_cost_hr = initial_reboiler_mw * self.steam_cost_per_mwh
        optimized_cost_hr = opt_reboiler * self.steam_cost_per_mwh
        hourly_savings = max(0.0, baseline_cost_hr - optimized_cost_hr)
        annual_savings = hourly_savings * self.annual_hours
        energy_reduction_pct = max(0.0, ((initial_reboiler_mw - opt_reboiler) / initial_reboiler_mw) * 100.0)

        return {
            "success": bool(res.success),
            "solver_message": res.message,
            "initial_setpoints": {
                "reflux_ratio": round(initial_reflux, 2),
                "reboiler_duty_mw": round(initial_reboiler_mw, 2),
                "hourly_cost_usd": round(baseline_cost_hr, 2),
            },
            "optimized_setpoints": {
                "reflux_ratio": round(float(opt_reflux), 2),
                "reboiler_duty_mw": round(float(opt_reboiler), 2),
                "hourly_cost_usd": round(optimized_cost_hr, 2),
            },
            "predicted_purity": round(float(optimized_purity), 4),
            "is_on_spec": bool(optimized_purity >= self.target_purity_min),
            "energy_reduction_pct": round(float(energy_reduction_pct), 2),
            "hourly_savings_usd": round(float(hourly_savings), 2),
            "annual_savings_usd": round(float(annual_savings), 2),
        }
