"""
AWS Lambda Handler for Real-time ETA Prediction
This module handles API requests and invokes SageMaker endpoint for predictions
"""

import json
import boto3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
dynamodb = boto3.resource('dynamodb')

# Configuration
SAGEMAKER_ENDPOINT = 'zomato-eta-predictor'
FEATURE_TABLE_NAME = 'eta-features'


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula
    """
    from math import radians, cos, sin, asin, sqrt
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r


def extract_time_features(timestamp: datetime) -> Dict[str, Any]:
    """
    Extract time-based features from timestamp
    """
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    
    features = {
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': 1 if day_of_week >= 5 else 0,
        'is_breakfast_rush': 1 if 7 <= hour <= 9 else 0,
        'is_lunch_rush': 1 if 12 <= hour <= 14 else 0,
        'is_dinner_rush': 1 if 19 <= hour <= 21 else 0,
        'is_late_night': 1 if hour >= 22 or hour <= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day_of_week / 7),
        'day_cos': np.cos(2 * np.pi * day_of_week / 7)
    }
    
    return features


def get_restaurant_features(restaurant_id: str) -> Dict[str, Any]:
    """
    Retrieve restaurant features from DynamoDB
    """
    try:
        table = dynamodb.Table('restaurants')
        response = table.get_item(Key={'restaurant_id': restaurant_id})
        
        if 'Item' in response:
            return {
                'restaurant_avg_prep_time': response['Item'].get('avg_prep_time', 20),
                'restaurant_rating': response['Item'].get('rating', 4.0),
                'restaurant_order_count': response['Item'].get('total_orders', 100)
            }
    except Exception as e:
        logger.error(f"Error fetching restaurant features: {str(e)}")
    
    # Return default values if fetch fails
    return {
        'restaurant_avg_prep_time': 20,
        'restaurant_rating': 4.0,
        'restaurant_order_count': 100
    }


def get_rider_features(rider_id: str) -> Dict[str, Any]:
    """
    Retrieve rider features from DynamoDB
    """
    try:
        table = dynamodb.Table('riders')
        response = table.get_item(Key={'rider_id': rider_id})
        
        if 'Item' in response:
            return {
                'rider_total_deliveries': response['Item'].get('total_deliveries', 50),
                'rider_rating': response['Item'].get('rating', 4.5),
                'vehicle_avg_speed_kmh': response['Item'].get('avg_speed', 30)
            }
    except Exception as e:
        logger.error(f"Error fetching rider features: {str(e)}")
    
    # Return default values if fetch fails
    return {
        'rider_total_deliveries': 50,
        'rider_rating': 4.5,
        'vehicle_avg_speed_kmh': 30
    }


def get_traffic_score(hour: int, is_weekend: int) -> float:
    """
    Estimate traffic score based on time
    """
    base_score = 0.5
    
    # Rush hour adjustments
    if 7 <= hour <= 9:
        base_score = 0.8
    elif 12 <= hour <= 14:
        base_score = 0.9
    elif 19 <= hour <= 21:
        base_score = 1.0
    elif hour >= 22 or hour <= 5:
        base_score = 0.2
    
    # Weekend adjustment
    if is_weekend:
        base_score *= 0.7
    
    return base_score


def prepare_features(request_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Prepare all features for model prediction
    
    Args:
        request_data: Request payload containing order details
        
    Returns:
        Dictionary of features for prediction
    """
    # Extract location data
    restaurant_lat = request_data['restaurant_lat']
    restaurant_lon = request_data['restaurant_lon']
    customer_lat = request_data['customer_lat']
    customer_lon = request_data['customer_lon']
    
    # Calculate distance
    distance_km = haversine_distance(
        restaurant_lat, restaurant_lon,
        customer_lat, customer_lon
    )
    
    # Get current timestamp
    timestamp = datetime.now()
    
    # Extract time features
    time_features = extract_time_features(timestamp)
    
    # Get restaurant features
    restaurant_features = get_restaurant_features(request_data.get('restaurant_id', 'default'))
    
    # Get rider features (if rider assigned)
    rider_features = {}
    if 'rider_id' in request_data:
        rider_features = get_rider_features(request_data['rider_id'])
    else:
        # Default rider features
        rider_features = {
            'rider_total_deliveries': 50,
            'rider_rating': 4.5,
            'vehicle_avg_speed_kmh': 30
        }
    
    # Get traffic score
    traffic_score = get_traffic_score(time_features['hour'], time_features['is_weekend'])
    
    # Weather features (simplified - in production, fetch from weather API)
    weather_delay_factor = request_data.get('weather_delay_factor', 1.0)
    
    # Combine all features
    features = {
        # Distance features
        'restaurant_to_customer_km': distance_km,
        'manhattan_distance_km': distance_km * 1.3,  # Approximation
        
        # Time features
        **time_features,
        
        # Restaurant features
        **restaurant_features,
        
        # Rider features
        **rider_features,
        
        # Traffic features
        'expected_traffic_score': traffic_score,
        'traffic_density_normalized': traffic_score,
        
        # Weather features
        'weather_delay_factor': weather_delay_factor,
        'is_raining': 1 if weather_delay_factor > 1.1 else 0,
        
        # Interaction features
        'distance_traffic_interaction': distance_km * traffic_score,
        'distance_weather_interaction': distance_km * weather_delay_factor,
        'prep_time_rush_interaction': restaurant_features['restaurant_avg_prep_time'] * 
                                      (time_features['is_lunch_rush'] + time_features['is_dinner_rush'])
    }
    
    return features


def invoke_sagemaker_endpoint(features: Dict[str, float]) -> float:
    """
    Invoke SageMaker endpoint for prediction
    
    Args:
        features: Feature dictionary
        
    Returns:
        Predicted ETA in minutes
    """
    try:
        # Convert features to list in correct order
        # Note: Feature order must match training data
        feature_list = list(features.values())
        
        # Prepare payload
        payload = json.dumps({'instances': [feature_list]})
        
        # Invoke endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        prediction = result['predictions'][0]
        
        return float(prediction)
        
    except Exception as e:
        logger.error(f"Error invoking SageMaker endpoint: {str(e)}")
        # Return fallback prediction based on distance
        return features['restaurant_to_customer_km'] * 3 + features['restaurant_avg_prep_time']


def calculate_delivery_time(eta_minutes: float) -> str:
    """
    Calculate estimated delivery time from current time
    
    Args:
        eta_minutes: Predicted ETA in minutes
        
    Returns:
        Formatted delivery time string
    """
    delivery_time = datetime.now() + timedelta(minutes=eta_minutes)
    return delivery_time.strftime('%I:%M %p')


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function for ETA prediction
    
    Args:
        event: API Gateway event
        context: Lambda context
        
    Returns:
        API response with ETA prediction
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Parse request body
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        # Validate required fields
        required_fields = ['restaurant_lat', 'restaurant_lon', 'customer_lat', 'customer_lon']
        for field in required_fields:
            if field not in body:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'error': f'Missing required field: {field}'
                    })
                }
        
        # Prepare features
        features = prepare_features(body)
        logger.info(f"Prepared features: {features}")
        
        # Get prediction
        eta_minutes = invoke_sagemaker_endpoint(features)
        
        # Add buffer (5-10% for safety)
        buffer_percentage = 0.07
        eta_with_buffer = eta_minutes * (1 + buffer_percentage)
        
        # Round to nearest minute
        eta_final = round(eta_with_buffer)
        
        # Calculate delivery time
        delivery_time = calculate_delivery_time(eta_final)
        
        # Prepare response
        response_body = {
            'eta_minutes': eta_final,
            'estimated_delivery_time': delivery_time,
            'distance_km': round(features['restaurant_to_customer_km'], 2),
            'prep_time_minutes': round(features['restaurant_avg_prep_time'], 0),
            'traffic_condition': 'high' if features['expected_traffic_score'] > 0.7 else 
                                'medium' if features['expected_traffic_score'] > 0.4 else 'low',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction successful: {response_body}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }


# For local testing
if __name__ == "__main__":
    # Sample test event
    test_event = {
        'body': json.dumps({
            'restaurant_lat': 28.6139,
            'restaurant_lon': 77.2090,
            'customer_lat': 28.6289,
            'customer_lon': 77.2190,
            'restaurant_id': 'REST001',
            'rider_id': 'RIDER001',
            'weather_delay_factor': 1.0
        })
    }
    
    response = lambda_handler(test_event, None)
    print(json.dumps(json.loads(response['body']), indent=2))
