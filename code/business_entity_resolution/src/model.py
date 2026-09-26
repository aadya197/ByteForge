from pathlib import Path
import joblib
import numpy as np
from xgboost import XGBClassifier

class MatchModel:
    def __init__(self, random_state=42):
        self.model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_columns = None

    def fit(self, X, y):
        self.feature_columns = list(X.columns)
        self.model.fit(X[self.feature_columns], y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.feature_columns])[:, 1]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
