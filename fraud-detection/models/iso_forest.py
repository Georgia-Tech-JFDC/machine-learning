import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, OutlierMixin
from typing import Optional


from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class TransactionIsoForestModel:
    def __init__(
        self,
        n_estimators=200,
        contamination="auto",
        max_samples="auto",
        random_state=42,
    ):
        """
        Skeleton for an Isolation Forest anomaly detection pipeline.
        """
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.fitted = False

    # ----------------------------------------------------------
    # STEP 1: Feature Engineering (placeholder)
    # ----------------------------------------------------------
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Insert your feature engineering here.
        df is raw transaction data.
        """

        # Example placeholder features:
        df = df.copy()
        df["log_amount"] = np.log1p(df["amount"])

        # TODO: Add real domain-specific features
        # df["hour"] = df["timestamp"].dt.hour
        # df["merchant_embedding"] = ...
        # df["rolling_avg"] = ...
        # df["user_baseline_deviation"] = ...
        # etc.

        # return only numeric features
        return df.select_dtypes(include=[np.number])

    # ----------------------------------------------------------
    # STEP 2: Fit Model
    # ----------------------------------------------------------
    def fit(self, df_raw: pd.DataFrame):
        """
        Train the Isolation Forest using raw transaction data.
        """

        X = self._preprocess_data(df_raw)
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)
        self.fitted = True
        return self

    # ----------------------------------------------------------
    # STEP 3: Predict Anomalies
    # ----------------------------------------------------------
    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Returns anomaly scores + labels.
        -1 = anomaly, 1 = normal
        """

        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._preprocess_data(df_raw)
        X_scaled = self.scaler.transform(X)

        preds = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        result = df_raw.copy()
        result["anomaly_label"] = preds
        result["anomaly_score"] = scores  # lower = more anomalous
        return result
