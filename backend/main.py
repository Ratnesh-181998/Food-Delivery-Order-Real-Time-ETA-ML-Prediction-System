from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from src.data.feature_engineering import FeatureEngineer

# Configure Logging
# Get absolute path for logs directory
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / "app.log"

# Create a custom logger
logger = logging.getLogger("zomato_eta_app")
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
# Check if handlers already exist to avoid duplicates
if not logger.handlers:
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

# Also add file handler to uvicorn logger to capture server logs
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.addHandler(f_handler)

# Initialize FastAPI app
app = FastAPI(
    title="Zomato ETA Prediction API",
    description="Real-time ML API for predicting food delivery times",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Zomato ETA Prediction Server Starting Up...")
    logger.info(f"📝 Logging to: {log_file}")

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"Path: {request.url.path} | Method: {request.method} | "
        f"Status: {response.status_code} | Duration: {process_time:.4f}s"
    )
    return response

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Feature Engineer
feature_engineer = FeatureEngineer()

# Mock model for demonstration if real model isn't trained yet
class MockModel:
    def predict(self, X):
        # Simple heuristic + random noise for demo
        base_time = 15
        distance_factor = X['restaurant_to_customer_km'] * 3
        prep_factor = X['restaurant_avg_prep_time'] * 0.8
        noise = np.random.normal(0, 3, len(X))
        return np.maximum(10, base_time + distance_factor + prep_factor + noise)

# Load Model
model = None
try:
    model_path = "models/xgboost_eta_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Loaded trained XGBoost model from {model_path}")
    else:
        print("Warning: Model not found. Using MockModel for demonstration.")
        model = MockModel()
except Exception as e:
    print(f"Error loading model: {e}")
    model = MockModel()

# Pydantic Models for Request/Response
class PredictionRequest(BaseModel):
    restaurant_lat: float
    restaurant_lon: float
    customer_lat: float
    customer_lon: float
    restaurant_name: Optional[str] = "Unknown"
    prep_time: Optional[int] = 20
    weather_condition: Optional[str] = "clear"
    order_time: Optional[str] = None

class PredictionResponse(BaseModel):
    eta_minutes: int
    estimated_delivery_time: str
    distance_km: float
    prep_time_minutes: int
    traffic_condition: str
    confidence_score: float
    features_used: Dict[str, Any]

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Zomato ETA Prediction API is running"}

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_eta(request: PredictionRequest):
    logger.info(f"Received prediction request for restaurant: {request.restaurant_name}")
    try:
        # 1. Prepare Data Frame
        data = {
            'restaurant_lat': [request.restaurant_lat],
            'restaurant_lon': [request.restaurant_lon],
            'customer_lat': [request.customer_lat],
            'customer_lon': [request.customer_lon],
            'prep_time_minutes': [request.prep_time],
            'weather_condition': [request.weather_condition],
            'order_timestamp': [request.order_time if request.order_time else datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'restaurant_id': ['REST_DEMO'],
            'restaurant_rating': [4.5]
        }
        df = pd.DataFrame(data)

        # 2. Feature Engineering
        df = feature_engineer.create_distance_features(df)
        df = feature_engineer.create_time_features(df)
        df = feature_engineer.create_restaurant_features(df)
        df = feature_engineer.create_traffic_features(df)
        df = feature_engineer.create_weather_features(df)
        df = feature_engineer.create_interaction_features(df)
        
        # Add missing columns if needed
        if 'is_raining' not in df.columns:
            df['is_raining'] = (df['weather_condition'].isin(['light_rain', 'heavy_rain'])).astype(int)
        if 'traffic_density_normalized' not in df.columns:
            df['traffic_density_normalized'] = df['expected_traffic_score']

        # 3. Select features in correct order for model
        feature_columns = [
            'restaurant_to_customer_km',
            'manhattan_distance_km',
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
            'restaurant_avg_prep_time',
            'restaurant_rating',
            'expected_traffic_score',
            'traffic_density_normalized',
            'weather_delay_factor',
            'is_raining',
            'distance_traffic_interaction',
            'distance_weather_interaction',
            'prep_time_rush_interaction'
        ]
        
        X = df[feature_columns]
        
        # 4. Predict
        prediction = model.predict(X)[0]
        eta_minutes = int(round(prediction))
        
        logger.info(f"Prediction successful: ETA {eta_minutes} mins for {request.restaurant_name}")

        # 5. Calculate Delivery Time
        arrival_time = datetime.now() + pd.Timedelta(minutes=eta_minutes)
        formatted_time = arrival_time.strftime("%I:%M %p")

        # 6. Determine Traffic Condition
        traffic_score = df['expected_traffic_score'].iloc[0]
        traffic_condition = 'high' if traffic_score > 0.7 else 'medium' if traffic_score > 0.4 else 'low'

        # 6. Construct Response
        return {
            "eta_minutes": eta_minutes,
            "estimated_delivery_time": formatted_time,
            "distance_km": round(df['restaurant_to_customer_km'].iloc[0], 2),
            "prep_time_minutes": request.prep_time,
            "traffic_condition": traffic_condition,
            "confidence_score": 0.89,
            "features_used": {
                "distance_km": round(df['restaurant_to_customer_km'].iloc[0], 2),
                "hour": int(df['hour'].iloc[0]),
                "is_rush_hour": bool(df['is_lunch_rush'].iloc[0] or df['is_dinner_rush'].iloc[0]),
                "weather": request.weather_condition,
                "traffic_score": round(traffic_score, 2)
            }
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (React build)
frontend_build_path = Path(__file__).parent.parent / "frontend" / "build"
if frontend_build_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_build_path / "static")), name="static")
    
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Serve index.html for all routes (React Router)
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        file_path = frontend_build_path / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        else:
            return FileResponse(frontend_build_path / "index.html")
else:
    print(f"Warning: Frontend build directory not found at {frontend_build_path}")
    print("Run 'npm run build' in the frontend directory to build the React app.")

if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("🚀 Starting Zomato ETA Prediction System")
    print("="*60)
    print(f"📊 Backend API: http://localhost:8001/api")
    print(f"🎨 Frontend UI: http://localhost:8001")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8001)
