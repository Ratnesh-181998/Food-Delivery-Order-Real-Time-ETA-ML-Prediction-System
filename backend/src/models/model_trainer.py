"""
Model Training Module for Zomato ETA Prediction
This module contains model training logic for various ML algorithms
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from typing import Tuple, Dict, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETAModelTrainer:
    """
    Model training class for ETA prediction
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'actual_delivery_time_minutes',
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training...")
        
        # Separate features and target
        X = df.drop(columns=[target_col, 'order_id'], errors='ignore')
        y = df[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Encoding {len(categorical_cols)} categorical columns")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> LinearRegression:
        """
        Train Linear Regression model (baseline)
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        logger.info("Training Linear Regression model...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        self.models['linear_regression'] = model
        logger.info("Linear Regression training complete")
        
        return model
    
    def train_ridge_regression(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        alpha: float = 1.0
    ) -> Ridge:
        """
        Train Ridge Regression model (L2 regularization)
        
        Args:
            X_train: Training features
            y_train: Training target
            alpha: Regularization strength
            
        Returns:
            Trained model
        """
        logger.info("Training Ridge Regression model...")
        
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        self.models['ridge_regression'] = model
        logger.info("Ridge Regression training complete")
        
        return model
    
    def train_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 20
    ) -> RandomForestRegressor:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest model...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        logger.info("Random Forest training complete")
        
        return model
    
    def train_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        params: Dict[str, Any] = None
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost model...")
        
        # Default parameters
        default_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        model = xgb.XGBRegressor(**default_params)
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=True
            )
        else:
            model.fit(X_train, y_train, verbose=True)
        
        self.models['xgboost'] = model
        logger.info("XGBoost training complete")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model
    
    def train_lightgbm(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        params: Dict[str, Any] = None
    ) -> lgb.LGBMRegressor:
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters
            
        Returns:
            Trained model
        """
        logger.info("Training LightGBM model...")
        
        # Default parameters
        default_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 7,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'regression',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        model = lgb.LGBMRegressor(**default_params)
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
        else:
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        logger.info("LightGBM training complete")
        
        return model
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"{model_name} Performance:")
        logger.info(f"  MAE: {mae:.2f} minutes")
        logger.info(f"  RMSE: {rmse:.2f} minutes")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series, 
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            
        Returns:
            Dictionary of model performances
        """
        logger.info("Training all models...")
        
        results = {}
        
        # Create validation set for early stopping
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )
        
        # Train baseline models
        lr_model = self.train_linear_regression(X_train, y_train)
        results['linear_regression'] = self.evaluate_model(lr_model, X_test, y_test, "Linear Regression")
        
        ridge_model = self.train_ridge_regression(X_train, y_train)
        results['ridge_regression'] = self.evaluate_model(ridge_model, X_test, y_test, "Ridge Regression")
        
        # Train ensemble models
        rf_model = self.train_random_forest(X_train, y_train)
        results['random_forest'] = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # Train gradient boosting models
        xgb_model = self.train_xgboost(X_train_sub, y_train_sub, X_val, y_val)
        results['xgboost'] = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        
        lgb_model = self.train_lightgbm(X_train_sub, y_train_sub, X_val, y_val)
        results['lightgbm'] = self.evaluate_model(lgb_model, X_test, y_test, "LightGBM")
        
        # Find best model based on MAE
        best_model_name = min(results, key=lambda x: results[x]['mae'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Best MAE: {results[best_model_name]['mae']:.2f} minutes")
        
        return results
    
    def save_model(self, model_name: str = None, output_path: str = "models/"):
        """
        Save trained model to disk
        
        Args:
            model_name: Name of model to save (default: best model)
            output_path: Directory to save model
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_path}{model_name}_{timestamp}.pkl"
        
        joblib.dump(model, filename)
        logger.info(f"Model saved to {filename}")
        
        # Save feature importance if available
        if self.feature_importance is not None:
            importance_file = f"{output_path}{model_name}_feature_importance_{timestamp}.csv"
            self.feature_importance.to_csv(importance_file, index=False)
            logger.info(f"Feature importance saved to {importance_file}")
    
    def load_model(self, model_path: str):
        """
        Load trained model from disk
        
        Args:
            model_path: Path to saved model
        """
        self.best_model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.best_model


def main():
    """
    Example usage of ETAModelTrainer
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'order_id': [f'ORD{i:04d}' for i in range(n_samples)],
        'restaurant_to_customer_km': np.random.uniform(1, 15, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'is_lunch_rush': np.random.randint(0, 2, n_samples),
        'is_dinner_rush': np.random.randint(0, 2, n_samples),
        'restaurant_avg_prep_time': np.random.uniform(10, 40, n_samples),
        'expected_traffic_score': np.random.uniform(0.2, 1.0, n_samples),
        'weather_delay_factor': np.random.uniform(1.0, 1.5, n_samples),
        'actual_delivery_time_minutes': np.random.uniform(20, 60, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize trainer
    trainer = ETAModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Train all models
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Save best model
    trainer.save_model()
    
    # Display results
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    results_df = pd.DataFrame(results).T
    print(results_df)


if __name__ == "__main__":
    main()
