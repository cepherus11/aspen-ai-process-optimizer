"""
Machine Learning Surrogate Model Suite
Trains, benchmarks, and evaluates regression surrogates (Linear, Ridge, Random Forest,
Gradient Boosting/XGBoost, MLP) and performs PCA dimensionality reduction.
"""

import numpy as np
import pandas as pd
import time
import pickle
import os
import logging
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

# Attempt to import XGBoost, fallback to GradientBoostingRegressor if unavailable
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class ProcessSurrogateSuite:
    """
    Trains and compares multiple machine learning algorithms as real-time surrogates
    for computationally intensive chemical process simulations.
    """

    FEATURE_COLS = [
        "feed_flow_kmol_hr",
        "feed_temp_c",
        "feed_c3_fraction",
        "reflux_ratio",
        "reboiler_duty_mw",
        "column_pressure_bar",
    ]

    TARGET_COLS = [
        "distillate_c3_purity",
        "bottoms_c3_loss",
        "condenser_duty_mw",
        "specific_energy_gj_ton",
        "hourly_operating_cost_usd",
    ]

    def __init__(self, primary_target: str = "distillate_c3_purity"):
        self.primary_target = primary_target
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.pca = PCA(n_components=3)
        self.models: Dict[str, Any] = {}
        self.benchmark_results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None

    def prepare_data(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts features and target, scales data, and splits into train/test sets.
        """
        X = df[self.FEATURE_COLS].values
        y = df[self.primary_target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def fit_pca(self, X_train_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Applies Principal Component Analysis (PCA) to uncover primary operational variance.
        """
        self.pca.fit(X_train_scaled)
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i+1}" for i in range(len(explained_variance))],
            index=self.FEATURE_COLS,
        )

        return {
            "explained_variance_ratio": explained_variance.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "loadings": loadings.to_dict(),
        }

    def train_and_benchmark(
        self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Trains and benchmarks:
        - Linear Regression (Baseline)
        - Ridge Regression (L2 Regularized)
        - Random Forest Regressor
        - Gradient Boosted Trees (XGBoost / GradientBoosting)
        - Multi-Layer Perceptron (ANN)
        """
        candidates = {
            "Linear_Regression": LinearRegression(),
            "Ridge_Regression": Ridge(alpha=1.0),
            "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
            "Multi_Layer_Perceptron": MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=400, random_state=42
            ),
        }

        if HAS_XGB:
            candidates["XGBoost_Regressor"] = xgb.XGBRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42
            )
        else:
            candidates["Gradient_Boosted_Trees"] = GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42
            )

        results = []

        for name, model in candidates.items():
            # Measure training time
            t0 = time.time()
            model.fit(X_train, y_train)
            train_duration = time.time() - t0

            # Measure inference latency (1,000 samples)
            t_inf_start = time.time()
            y_pred = model.predict(X_test)
            inference_duration_ms = ((time.time() - t_inf_start) / len(X_test)) * 1000.0

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            self.models[name] = model
            self.benchmark_results[name] = {
                "r2_score": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "train_time_sec": float(train_duration),
                "inference_latency_ms": float(inference_duration_ms),
            }

            results.append(
                {
                    "Model": name,
                    "R2_Score": round(r2, 4),
                    "RMSE": round(rmse, 6),
                    "MAE": round(mae, 6),
                    "Inference_Latency_ms": round(inference_duration_ms, 4),
                    "Train_Time_s": round(train_duration, 3),
                }
            )

        results_df = pd.DataFrame(results).sort_values(by="R2_Score", ascending=False)
        self.best_model_name = results_df.iloc[0]["Model"]
        self.best_model = self.models[self.best_model_name]

        return results_df

    def predict_single(self, input_dict: Dict[str, float]) -> float:
        """
        Fast single-instance surrogate prediction (< 1 millisecond).
        """
        if self.best_model is None:
            raise ValueError("Model is not trained yet.")

        feature_vector = np.array([[input_dict[f] for f in self.FEATURE_COLS]])
        scaled_vector = self.scaler_X.transform(feature_vector)
        pred = self.best_model.predict(scaled_vector)[0]
        return float(pred)

    def save(self, filepath: str):
        """Serializes trained surrogate suite."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "primary_target": self.primary_target,
                    "scaler_X": self.scaler_X,
                    "pca": self.pca,
                    "models": self.models,
                    "best_model_name": self.best_model_name,
                    "benchmark_results": self.benchmark_results,
                },
                f,
            )

    @classmethod
    def load(cls, filepath: str) -> "ProcessSurrogateSuite":
        """Loads serialized surrogate suite."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        suite = cls(primary_target=data["primary_target"])
        suite.scaler_X = data["scaler_X"]
        suite.pca = data["pca"]
        suite.models = data["models"]
        suite.best_model_name = data["best_model_name"]
        suite.best_model = data["models"][data["best_model_name"]]
        suite.benchmark_results = data["benchmark_results"]
        return suite
