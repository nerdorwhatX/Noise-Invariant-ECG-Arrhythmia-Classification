import os
import joblib
import logging
import lightgbm as lgb


logger = logging.getLogger(__name__)


class HybridLGBMClassifier:
    """
    Our main LightGBM model. It uses the 89 features we extracted (DSP, Wavelet, TDA, etc.)
    We're using balanced class weights since the V and S classes are rare.
    """

    def __init__(
        self, n_estimators=1000, learning_rate=0.05, max_depth=-1, random_state=42
    ):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            class_weight="balanced",  # Crucial for the rare S and F classes
            subsample=0.8,
            colsample_bytree=0.8,
            importance_type="gain",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM classifier with early stopping if validation data is provided.
        """
        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=True))
            callbacks.append(lgb.log_evaluation(period=50))

        logger.info("Training LightGBM on engineered features...")
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

    def get_feature_importances(self):
        """Returns the sorted feature importances (gain)."""
        return self.model.feature_importances_
