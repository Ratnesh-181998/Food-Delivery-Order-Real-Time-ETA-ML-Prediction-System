"""
Generate dummy training data for Zomato ETA Prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate 10,000 training samples
n_samples = 10000

print(f"Generating {n_samples} training samples...")

# Delhi NCR coordinates range
delhi_lat_range = (28.4, 28.9)
delhi_lon_range = (76.8, 77.4)

# Generate data
data = {
    # Location features
    'restaurant_lat': np.random.uniform(delhi_lat_range[0], delhi_lat_range[1], n_samples),
    'restaurant_lon': np.random.uniform(delhi_lon_range[0], delhi_lon_range[1], n_samples),
    'customer_lat': np.random.uniform(delhi_lat_range[0], delhi_lat_range[1], n_samples),
    'customer_lon': np.random.uniform(delhi_lon_range[0], delhi_lon_range[1], n_samples),
    
    # Restaurant features
    'restaurant_id': [f'REST_{i%500:04d}' for i in range(n_samples)],
    'restaurant_rating': np.random.uniform(3.5, 5.0, n_samples),
    'prep_time_minutes': np.random.randint(10, 40, n_samples),
    
    # Time features
    'order_timestamp': [
        (datetime.now() - timedelta(days=random.randint(0, 90), 
                                    hours=random.randint(0, 23),
                                    minutes=random.randint(0, 59)))
        .strftime("%Y-%m-%d %H:%M:%S")
        for _ in range(n_samples)
    ],
    
    # Weather features
    'weather_condition': np.random.choice(['clear', 'cloudy', 'light_rain', 'heavy_rain'], 
                                         n_samples, 
                                         p=[0.6, 0.25, 0.1, 0.05]),
    
    # Rider features
    'rider_id': [f'RIDER_{i%200:04d}' for i in range(n_samples)],
    'rider_total_deliveries': np.random.randint(50, 1000, n_samples),
    'rider_rating': np.random.uniform(4.0, 5.0, n_samples),
}

df = pd.DataFrame(data)

# Calculate actual ETA (target variable) based on realistic factors
def calculate_realistic_eta(row):
    # Haversine distance calculation
    from math import radians, cos, sin, asin, sqrt
    
    lon1, lat1, lon2, lat2 = map(radians, [row['restaurant_lon'], row['restaurant_lat'], 
                                            row['customer_lon'], row['customer_lat']])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    distance_km = c * 6371
    
    # Parse timestamp
    timestamp = datetime.strptime(row['order_timestamp'], "%Y-%m-%d %H:%M:%S")
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    
    # Base time from distance (avg speed 20-30 km/h depending on traffic)
    is_rush_hour = hour in [8, 9, 12, 13, 19, 20, 21]
    is_weekend = day_of_week >= 5
    
    avg_speed = 15 if is_rush_hour else 25 if not is_weekend else 22
    travel_time = (distance_km / avg_speed) * 60  # Convert to minutes
    
    # Add prep time
    prep_time = row['prep_time_minutes']
    
    # Weather impact
    weather_delays = {
        'clear': 0,
        'cloudy': 2,
        'light_rain': 5,
        'heavy_rain': 10
    }
    weather_delay = weather_delays.get(row['weather_condition'], 0)
    
    # Restaurant rating impact (better restaurants are faster)
    restaurant_factor = (5.0 - row['restaurant_rating']) * 2
    
    # Rider experience impact
    rider_factor = max(0, (500 - row['rider_total_deliveries']) / 100)
    
    # Calculate total ETA
    eta = prep_time + travel_time + weather_delay + restaurant_factor + rider_factor
    
    # Add some random noise
    noise = np.random.normal(0, 3)
    
    return max(10, eta + noise)  # Minimum 10 minutes

print("Calculating realistic ETA values...")
df['actual_eta_minutes'] = df.apply(calculate_realistic_eta, axis=1)

# Save to CSV
output_path = 'data/training_data.csv'
df.to_csv(output_path, index=False)

print(f"\n✅ Generated {n_samples} samples")
print(f"📁 Saved to: {output_path}")
print(f"\n📊 Data Statistics:")
print(f"   - Mean ETA: {df['actual_eta_minutes'].mean():.2f} minutes")
print(f"   - Median ETA: {df['actual_eta_minutes'].median():.2f} minutes")
print(f"   - Min ETA: {df['actual_eta_minutes'].min():.2f} minutes")
print(f"   - Max ETA: {df['actual_eta_minutes'].max():.2f} minutes")
print(f"   - Std Dev: {df['actual_eta_minutes'].std():.2f} minutes")

print("\n✅ Data generation complete!")
