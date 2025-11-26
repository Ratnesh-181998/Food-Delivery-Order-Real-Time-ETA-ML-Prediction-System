"""
Feature Engineering Module for Zomato ETA Prediction
This module contains all feature engineering logic including distance calculations,
time-based features, and categorical encodings.
"""

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for ETA prediction system
    """
    
    def __init__(self):
        self.feature_names = []
        
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on the earth
        using the Haversine formula (accounts for Earth's curvature)
        
        Args:
            lat1, lon1: Latitude and longitude of point 1 (in decimal degrees)
            lat2, lon2: Latitude and longitude of point 2 (in decimal degrees)
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    @staticmethod
    def manhattan_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate Manhattan distance in kilometers (useful for city grid layouts)
        
        Args:
            lat1, lon1: Latitude and longitude of point 1
            lat2, lon2: Latitude and longitude of point 2
            
        Returns:
            Manhattan distance in kilometers
        """
        # Approximate conversion: 1 degree latitude ≈ 111 km
        lat_diff_km = abs(lat2 - lat1) * 111
        
        # Longitude distance varies with latitude
        avg_lat = (lat1 + lat2) / 2
        lon_diff_km = abs(lon2 - lon1) * 111 * cos(radians(avg_lat))
        
        return lat_diff_km + lon_diff_km
    
    def create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create distance-based features
        
        Args:
            df: DataFrame with location columns
            
        Returns:
            DataFrame with additional distance features
        """
        logger.info("Creating distance features...")
        
        # Restaurant to customer distance (Haversine)
        df['restaurant_to_customer_km'] = df.apply(
            lambda row: self.haversine_distance(
                row['restaurant_lat'], row['restaurant_lon'],
                row['customer_lat'], row['customer_lon']
            ),
            axis=1
        )
        
        # Rider to restaurant distance (if rider location available)
        if 'rider_lat' in df.columns:
            df['rider_to_restaurant_km'] = df.apply(
                lambda row: self.haversine_distance(
                    row['rider_lat'], row['rider_lon'],
                    row['restaurant_lat'], row['restaurant_lon']
                ),
                axis=1
            )
            
            # Total delivery distance
            df['total_delivery_distance_km'] = (
                df['rider_to_restaurant_km'] + df['restaurant_to_customer_km']
            )
        
        # Manhattan distance (for urban areas with grid layouts)
        df['manhattan_distance_km'] = df.apply(
            lambda row: self.manhattan_distance_km(
                row['restaurant_lat'], row['restaurant_lon'],
                row['customer_lat'], row['customer_lon']
            ),
            axis=1
        )
        
        logger.info(f"Created {4} distance features")
        return df
    
    def create_time_features(self, df: pd.DataFrame, timestamp_col: str = 'order_timestamp') -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of the timestamp column
            
        Returns:
            DataFrame with additional time features
        """
        logger.info("Creating time features...")
        
        # Convert to datetime if not already
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Rush hour indicators
        df['is_breakfast_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_lunch_rush'] = ((df['hour'] >= 12) & (df['hour'] <= 14)).astype(int)
        df['is_dinner_rush'] = ((df['hour'] >= 19) & (df['hour'] <= 21)).astype(int)
        df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Cyclical encoding for hour (preserves circular nature)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info(f"Created {14} time features")
        return df
    
    def create_restaurant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create restaurant-specific features
        
        Args:
            df: DataFrame with restaurant information
            
        Returns:
            DataFrame with additional restaurant features
        """
        logger.info("Creating restaurant features...")
        
        # Average preparation time by restaurant
        if 'prep_time_minutes' in df.columns:
            restaurant_avg_prep = df.groupby('restaurant_id')['prep_time_minutes'].transform('mean')
            df['restaurant_avg_prep_time'] = restaurant_avg_prep
            
            # Deviation from average
            df['prep_time_deviation'] = df['prep_time_minutes'] - df['restaurant_avg_prep_time']
        
        # Restaurant popularity (order count)
        restaurant_order_count = df.groupby('restaurant_id').size()
        df['restaurant_order_count'] = df['restaurant_id'].map(restaurant_order_count)
        
        # Restaurant rating features
        if 'restaurant_rating' in df.columns:
            df['is_high_rated'] = (df['restaurant_rating'] >= 4.0).astype(int)
        
        # Cuisine type encoding (will be handled separately with embeddings)
        
        logger.info("Created restaurant features")
        return df
    
    def create_rider_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rider-specific features
        
        Args:
            df: DataFrame with rider information
            
        Returns:
            DataFrame with additional rider features
        """
        logger.info("Creating rider features...")
        
        # Rider experience (total deliveries)
        if 'rider_id' in df.columns:
            rider_delivery_count = df.groupby('rider_id').size()
            df['rider_total_deliveries'] = df['rider_id'].map(rider_delivery_count)
            
            # Experience level
            df['rider_experience_level'] = pd.cut(
                df['rider_total_deliveries'],
                bins=[0, 100, 500, 1000, float('inf')],
                labels=['novice', 'intermediate', 'experienced', 'expert']
            )
        
        # Rider rating
        if 'rider_rating' in df.columns:
            df['is_top_rider'] = (df['rider_rating'] >= 4.5).astype(int)
        
        # Vehicle type encoding
        if 'vehicle_type' in df.columns:
            vehicle_speed_map = {
                'bicycle': 15,
                'scooter': 30,
                'motorcycle': 40,
                'car': 35
            }
            df['vehicle_avg_speed_kmh'] = df['vehicle_type'].map(vehicle_speed_map)
        
        logger.info("Created rider features")
        return df
    
    def create_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create traffic-related features
        
        Args:
            df: DataFrame with traffic information
            
        Returns:
            DataFrame with additional traffic features
        """
        logger.info("Creating traffic features...")
        
        # Traffic density score (if available from external API)
        if 'traffic_density' in df.columns:
            # Normalize traffic density
            df['traffic_density_normalized'] = (
                df['traffic_density'] - df['traffic_density'].min()
            ) / (df['traffic_density'].max() - df['traffic_density'].min())
        
        # Expected traffic based on time
        df['expected_traffic_score'] = 0.5  # Default medium traffic
        
        # High traffic during rush hours
        df.loc[df['is_breakfast_rush'] == 1, 'expected_traffic_score'] = 0.8
        df.loc[df['is_lunch_rush'] == 1, 'expected_traffic_score'] = 0.9
        df.loc[df['is_dinner_rush'] == 1, 'expected_traffic_score'] = 1.0
        df.loc[df['is_late_night'] == 1, 'expected_traffic_score'] = 0.2
        
        # Weekend adjustment (less traffic on weekends)
        df.loc[df['is_weekend'] == 1, 'expected_traffic_score'] *= 0.7
        
        logger.info("Created traffic features")
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features
        
        Args:
            df: DataFrame with weather information
            
        Returns:
            DataFrame with additional weather features
        """
        logger.info("Creating weather features...")
        
        # Weather impact on delivery
        if 'weather_condition' in df.columns:
            weather_impact_map = {
                'clear': 1.0,
                'cloudy': 1.0,
                'light_rain': 1.2,
                'heavy_rain': 1.5,
                'storm': 2.0,
                'fog': 1.3
            }
            df['weather_delay_factor'] = df['weather_condition'].map(weather_impact_map).fillna(1.0)
        
        # Temperature impact
        if 'temperature_celsius' in df.columns:
            # Extreme temperatures may slow down delivery
            df['is_extreme_temp'] = (
                (df['temperature_celsius'] < 5) | (df['temperature_celsius'] > 40)
            ).astype(int)
        
        # Precipitation
        if 'precipitation_mm' in df.columns:
            df['is_raining'] = (df['precipitation_mm'] > 0).astype(int)
            df['rain_intensity'] = pd.cut(
                df['precipitation_mm'],
                bins=[0, 2.5, 7.5, float('inf')],
                labels=['no_rain', 'light', 'heavy']
            )
        
        logger.info("Created weather features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        # Distance × Traffic
        if 'restaurant_to_customer_km' in df.columns and 'expected_traffic_score' in df.columns:
            df['distance_traffic_interaction'] = (
                df['restaurant_to_customer_km'] * df['expected_traffic_score']
            )
        
        # Distance × Weather
        if 'restaurant_to_customer_km' in df.columns and 'weather_delay_factor' in df.columns:
            df['distance_weather_interaction'] = (
                df['restaurant_to_customer_km'] * df['weather_delay_factor']
            )
        
        # Prep time × Rush hour
        if 'restaurant_avg_prep_time' in df.columns:
            df['prep_time_rush_interaction'] = (
                df['restaurant_avg_prep_time'] * 
                (df['is_lunch_rush'] + df['is_dinner_rush'])
            )
        
        logger.info("Created interaction features")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting complete feature engineering pipeline...")
        
        # Create all feature groups
        df = self.create_distance_features(df)
        df = self.create_time_features(df)
        df = self.create_restaurant_features(df)
        df = self.create_rider_features(df)
        df = self.create_traffic_features(df)
        df = self.create_weather_features(df)
        df = self.create_interaction_features(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df


def main():
    """
    Example usage of FeatureEngineer class
    """
    # Sample data
    sample_data = {
        'order_id': ['ORD001', 'ORD002', 'ORD003'],
        'restaurant_lat': [28.6139, 28.5355, 28.7041],
        'restaurant_lon': [77.2090, 77.3910, 77.1025],
        'customer_lat': [28.6289, 28.5505, 28.7191],
        'customer_lon': [77.2190, 77.4010, 77.1125],
        'order_timestamp': ['2024-01-15 12:30:00', '2024-01-15 19:45:00', '2024-01-15 08:15:00'],
        'restaurant_id': ['REST001', 'REST002', 'REST001'],
        'prep_time_minutes': [20, 25, 18],
        'restaurant_rating': [4.5, 4.2, 4.5],
        'weather_condition': ['clear', 'light_rain', 'clear'],
        'temperature_celsius': [25, 22, 20]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Engineer features
    df_engineered = fe.engineer_all_features(df)
    
    # Display results
    print("\nEngineered Features:")
    print(df_engineered.head())
    print(f"\nTotal columns: {len(df_engineered.columns)}")
    print(f"Column names: {list(df_engineered.columns)}")


if __name__ == "__main__":
    main()
