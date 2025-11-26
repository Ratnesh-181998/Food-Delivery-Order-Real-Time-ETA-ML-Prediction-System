"""
Train XGBoost model for ETA prediction
"""
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

from src.data.feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

print("="*60)
print("   Zomato ETA Prediction - Model Training")
print("="*60)

# Load training data
print("\n📂 Loading training data...")
data_path = 'data/training_data.csv'
df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df)} samples")

# Initialize feature engineer
print("\n⚙️  Performing feature engineering...")
feature_engineer = FeatureEngineer()

# Apply feature engineering
df = feature_engineer.create_distance_features(df)
df = feature_engineer.create_time_features(df)
df = feature_engineer.create_restaurant_features(df)
# Skip rider features for now since we don't have rider location data
df = feature_engineer.create_traffic_features(df)
df = feature_engineer.create_weather_features(df)
df = feature_engineer.create_interaction_features(df)

# Add is_raining if not present
if 'is_raining' not in df.columns:
    df['is_raining'] = (df['weather_condition'].isin(['light_rain', 'heavy_rain'])).astype(int)

# Add traffic_density_normalized if not present
if 'traffic_density_normalized' not in df.columns:
    df['traffic_density_normalized'] = df['expected_traffic_score']

print(f"✅ Created {len(df.columns)} features")

# Select features for training
feature_columns = [
    # Distance features
    'restaurant_to_customer_km',
    'manhattan_distance_km',
    
    # Time features
    'hour',
    'day_of_week',
    'is_weekend',
    'is_breakfast_rush',
    'is_lunch_rush',
    'is_dinner_rush',
    'is_late_night',
    'hour_sin',
    'hour_cos',
    'day_sin',
    'day_cos',
    
    # Restaurant features
    'restaurant_avg_prep_time',
    'restaurant_rating',
    
    # Traffic features
    'expected_traffic_score',
    'traffic_density_normalized',
    
    # Weather features
    'weather_delay_factor',
    'is_raining',
    
    # Interaction features
    'distance_traffic_interaction',
    'distance_weather_interaction',
    'prep_time_rush_interaction'
]

# Prepare X and y
X = df[feature_columns]
y = df['actual_eta_minutes']

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Target variable shape: {y.shape}")

# Split data
print("\n✂️  Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   - Training samples: {len(X_train)}")
print(f"   - Test samples: {len(X_test)}")

# Train XGBoost model
print("\n🤖 Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric='mae'
)

# Train with validation set
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

print("\n✅ Model training complete!")

# Evaluate model
print("\n📈 Evaluating model performance...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Training metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

# Test metrics
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print("\n" + "="*60)
print("   MODEL PERFORMANCE")
print("="*60)
print("\n📊 Training Set:")
print(f"   - MAE:  {train_mae:.2f} minutes")
print(f"   - RMSE: {train_rmse:.2f} minutes")
print(f"   - R²:   {train_r2:.4f}")

print("\n📊 Test Set:")
print(f"   - MAE:  {test_mae:.2f} minutes")
print(f"   - RMSE: {test_rmse:.2f} minutes")
print(f"   - R²:   {test_r2:.4f}")
print(f"   - MAPE: {test_mape:.2f}%")

# Feature importance
print("\n⭐ Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:35s} {row['importance']:.4f}")

# Save model
print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)
model_path = 'models/xgboost_eta_model.pkl'
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")

# Save feature columns for inference
feature_info = {
    'feature_columns': feature_columns,
    'model_type': 'XGBoost',
    'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'metrics': {
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'test_mape': float(test_mape)
    }
}

import json
with open('models/model_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("✅ Model info saved to: models/model_info.json")

print("\n" + "="*60)
print("   ✅ TRAINING COMPLETE!")
print("="*60)
print(f"\n🎯 Model achieves {test_mae:.2f} min MAE on test set")
print(f"🎯 R² Score: {test_r2:.4f}")
print("\n💡 To use this model, update backend/main.py:")
print("   model_path = 'models/xgboost_eta_model.pkl'")
