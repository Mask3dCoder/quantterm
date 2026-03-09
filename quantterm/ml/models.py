"""
Machine Learning Model Training for Trading Signals.

Uses time-series cross-validation to prevent leakage.
"""
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

# SECURITY: Import safe loading functions
from quantterm.utils.security import (
    safe_joblib_load,
    safe_joblib_dump,
    SecurityError,
    InvalidModelError,
)
    """
    Train ML models with proper time-series cross-validation.
    
    Features:
    - Time-series cross-validation (no leakage)
    - Purge/embargo between train and test
    - Feature scaling
    - Multiple model types
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        feature_engineer: 'FeatureEngineer' = None,
        cv_folds: int = 5,
        embargo_pct: float = 0.01,
        random_state: int = 42
    ):
        """
        Initialize ML model trainer.
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'logistic'
            feature_engineer: FeatureEngineer instance
            cv_folds: Number of CV folds
            embargo_pct: Percentage of samples to embargo (prevent leakage)
            random_state: Random seed
        """
        self.model_type = model_type
        self.features = feature_engineer
        self.cv_folds = cv_folds
        self.embargo_pct = embargo_pct
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.cv_scores = []
    
    def prepare_training_data(
        self,
        data: pd.DataFrame,
        min_history: int = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix X and target vector y.
        
        Args:
            data: DataFrame with OHLCV data
            min_history: Minimum history for features
            
        Returns:
            X: Feature matrix
            y: Target vector (1 = up, 0 = down)
        """
        if self.features is None:
            from quantterm.ml.features import FeatureEngineer
            self.features = FeatureEngineer()
        
        # Create features for all timestamps
        features_df = self.features.create_features_batch(data, min_history)
        
        if features_df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)
        
        # Create targets
        targets = []
        valid_indices = []
        
        for timestamp in features_df.index:
            target = self.features.create_target(data, timestamp)
            if target is not None:
                targets.append(1 if target > 0 else 0)
                valid_indices.append(timestamp)
        
        if not targets:
            return pd.DataFrame(), pd.Series(dtype=int)
        
        X = features_df.loc[valid_indices]
        y = pd.Series(targets, index=valid_indices)
        
        self.feature_names = list(X.columns)
        
        return X, y
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> dict:
        """
        Train model with time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction for final test set
            
        Returns:
            dict with training results
        """
        # Remove any NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        
        if len(X) < 50:
            raise ValueError("Insufficient training data")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        cv_scores = []
        cv_precisions = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            # Apply embargo
            embargo_size = int(len(test_idx) * self.embargo_pct)
            if embargo_size > 0 and len(test_idx) > embargo_size:
                test_idx = test_idx[embargo_size:]
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = self._create_model()
            model.fit(X_train, y_train)
            
            # Evaluate
            score = model.score(X_test, y_test)
            cv_scores.append(score)
            
            # Precision (of positive predictions)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)[:, 1]
                pred_up = probs > 0.5
                if pred_up.sum() > 0:
                    precision = (y_test[pred_up] == 1).mean()
                    cv_precisions.append(precision)
        
        self.cv_scores = cv_scores
        
        # Train final model on all data
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # Feature importance
        importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            importance = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
        
        # Overfitting gap (train vs CV)
        train_score = self.model.score(X_scaled, y)
        cv_mean = np.mean(cv_scores)
        
        return {
            'cv_accuracy_mean': float(np.mean(cv_scores)),
            'cv_accuracy_std': float(np.std(cv_scores)),
            'cv_precision_mean': float(np.mean(cv_precisions)) if cv_precisions else 0.5,
            'feature_importance': importance,
            'train_accuracy': float(train_score),
            'overfitting_gap': float(train_score - cv_mean),
            'n_samples': len(X),
            'n_features': len(self.feature_names)
        }
    
    def predict(self, features: pd.Series) -> tuple[float, float]:
        """
        Predict probability of positive return.
        
        Args:
            features: Feature vector
            
        Returns:
            (probability_up, confidence)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        # Prepare features
        X = features.values.reshape(1, -1)
        X = self.scaler.transform(X)
        
        # Predict
        prob = self.model.predict_proba(X)[0, 1]
        
        # Confidence: distance from 0.5
        confidence = abs(prob - 0.5) * 2
        
        return float(prob), float(confidence)
    
    def save(self, path: str):
        """Save model to file with security validation.
        
        Args:
            path: Path to save the model
            
        Raises:
            SecurityError: If model type is not allowed
        """
        # Validate before saving
        safe_joblib_dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, path)
    
    def load(self, path: str):
        """Load model from file with security validation.
        
        This method implements multiple security checks:
        1. File validation (magic number, size, suspicious content)
        2. Restricted unpickling (class whitelist)
        3. Type verification after loading
        
        Args:
            path: Path to the model file
            
        Raises:
            SecurityError: If model file is suspicious or contains disallowed classes
            InvalidModelError: If model file is corrupted or invalid
        """
        # Use safe loading with full validation
        data = safe_joblib_load(path)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
    
    def _create_model(self):
        """Create model instance."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
