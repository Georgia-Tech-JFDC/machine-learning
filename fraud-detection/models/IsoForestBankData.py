import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class BankTransactionIsoForestModel:
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
    # STEP 1: Feature Engineering
    # ----------------------------------------------------------
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'TransactionDate' in df.columns:
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
        
        if 'PreviousTransactionDate' in df.columns:
            df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'], errors='coerce')
        
        if 'TransactionDate' in df.columns:
            df['transaction_hour'] = df['TransactionDate'].dt.hour
            df['transaction_day_of_week'] = df['TransactionDate'].dt.dayofweek
            df['transaction_month'] = df['TransactionDate'].dt.month
            df['transaction_day'] = df['TransactionDate'].dt.day
            
            # Calculate time since previous transaction (in hours)
            if 'PreviousTransactionDate' in df.columns:
                df['hours_since_previous'] = (
                    (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds() / 3600
                )
                df['hours_since_previous'] = df['hours_since_previous'].fillna(24 * 30)
        
        if 'AccountID' in df.columns:
            account_counts = df['AccountID'].value_counts().to_dict()
            df['account_transaction_count'] = df['AccountID'].map(account_counts)
        
        categorical_cols = ['TransactionType', 'Channel', 'Location', 'CustomerOccupation']
        
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        if 'TransactionAmount' in df.columns:
            df['log_transaction_amount'] = np.log1p(df['TransactionAmount'])
        
        if 'AccountBalance' in df.columns:
            df['log_account_balance'] = np.log1p(df['AccountBalance'].clip(lower=0))
        
       
        numeric_features = []
        
        if 'TransactionAmount' in df.columns:
            numeric_features.append('TransactionAmount')
        if 'log_transaction_amount' in df.columns:
            numeric_features.append('log_transaction_amount')
        
        if 'AccountBalance' in df.columns:
            numeric_features.append('AccountBalance')
        if 'log_account_balance' in df.columns:
            numeric_features.append('log_account_balance')
        
        if 'CustomerAge' in df.columns:
            numeric_features.append('CustomerAge')
        
        if 'TransactionDuration' in df.columns:
            numeric_features.append('TransactionDuration')
        
        if 'LoginAttempts' in df.columns:
            numeric_features.append('LoginAttempts')
        
        time_features = ['transaction_hour', 'transaction_day_of_week', 
                        'transaction_month', 'transaction_day', 'hours_since_previous']
        for feat in time_features:
            if feat in df.columns:
                numeric_features.append(feat)
        
        if 'account_transaction_count' in df.columns:
            numeric_features.append('account_transaction_count')
        
        for col in df.columns:
            if any(col.startswith(cat_col + '_') for cat_col in categorical_cols):
                numeric_features.append(col)
        
        available_features = [f for f in numeric_features if f in df.columns]
        X = df[available_features].copy()
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X = X.drop(columns=[col])
                    continue
        
        for col in X.columns:
            if X[col].isna().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
        
        X = X.select_dtypes(include=[np.number])
        
        return X

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
        result["anomaly_score"] = scores  
        return result