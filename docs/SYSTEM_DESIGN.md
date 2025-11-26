# System Design Document: Zomato ETA Prediction

## Executive Summary

This document outlines the comprehensive machine learning system design for predicting Estimated Time of Arrival (ETA) in Zomato's food delivery platform. The system leverages AWS cloud services and advanced ML algorithms to provide real-time, accurate delivery time predictions.

## 1. Problem Statement

### 1.1 Business Objective
Provide customers with accurate, real-time estimates of when their food will arrive, considering multiple dynamic factors including:
- Restaurant preparation time
- Rider location and availability
- Traffic conditions
- Weather conditions
- Historical delivery patterns

### 1.2 Success Metrics
- **Accuracy**: Mean Absolute Error (MAE) < 5 minutes
- **Latency**: Prediction response time < 100ms
- **Availability**: 99.99% uptime
- **Customer Satisfaction**: Reduce ETA variance complaints by 40%

## 2. System Requirements

### 2.1 Functional Requirements

| Requirement | Description | Priority |
|-------------|-------------|----------|
| FR-1 | Real-time ETA prediction for active orders | Critical |
| FR-2 | Dynamic ETA updates based on real-time conditions | Critical |
| FR-3 | Integration with restaurant management system | High |
| FR-4 | Integration with rider tracking system | High |
| FR-5 | Historical data analysis and reporting | Medium |
| FR-6 | A/B testing framework for model variants | Medium |

### 2.2 Non-Functional Requirements

| Requirement | Description | Target |
|-------------|-------------|--------|
| NFR-1 | Scalability | Handle 1M+ concurrent predictions |
| NFR-2 | Latency | P99 < 100ms |
| NFR-3 | Availability | 99.99% uptime |
| NFR-4 | Data Freshness | Real-time features < 1 second old |
| NFR-5 | Model Accuracy | MAE < 5 minutes |
| NFR-6 | Cost Efficiency | < $0.001 per prediction |

## 3. Data Architecture

### 3.1 Data Sources

#### Primary Data Sources
1. **Order Management System**
   - Order details (ID, timestamp, items, value)
   - Customer information (ID, location, preferences)
   - Payment status

2. **Restaurant Management System**
   - Restaurant details (ID, location, cuisine)
   - Menu and preparation times
   - Current queue length
   - Operating hours and capacity

3. **Rider Management System**
   - Rider details (ID, vehicle type, rating)
   - Real-time GPS location
   - Current assignment status
   - Historical performance metrics

#### External Data Sources
1. **Traffic Data**
   - Google Maps Traffic API
   - Real-time traffic density
   - Expected travel time

2. **Weather Data**
   - OpenWeather API
   - Current conditions
   - Precipitation forecast

3. **Events & Holidays**
   - Public holiday calendar
   - Local events affecting traffic

### 3.2 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Orders  │  │Restaurant│  │  Riders  │  │ External │   │
│  │   DB     │  │    DB    │  │   DB     │  │   APIs   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │      AWS Kinesis Data Streams       │
        │     (Real-time Data Ingestion)      │
        └─────────────────┬───────────────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
    ┌───────────────────┐  ┌──────────────────┐
    │   AWS Kinesis     │  │    AWS S3        │
    │  Data Firehose    │  │  (Data Lake)     │
    │  (Streaming ETL)  │  │  - Raw Data      │
    └─────────┬─────────┘  │  - Processed     │
              │            │  - Features      │
              └────────────┤  - Models        │
                          └────────┬──────────┘
                                   │
                          ┌────────┴──────────┐
                          │    AWS Glue       │
                          │  (Batch ETL)      │
                          └────────┬──────────┘
                                   │
                          ┌────────┴──────────┐
                          │  Feature Store    │
                          │  (DynamoDB +      │
                          │   S3 Feature      │
                          │   Store)          │
                          └───────────────────┘
```

### 3.3 Data Storage Strategy

| Data Type | Storage | Retention | Purpose |
|-----------|---------|-----------|---------|
| Raw Events | S3 (Parquet) | 2 years | Audit, retraining |
| Processed Features | S3 + DynamoDB | 90 days | Model training |
| Real-time Features | DynamoDB | 7 days | Online inference |
| Model Artifacts | S3 | Indefinite | Model versioning |
| Predictions | DynamoDB + S3 | 1 year | Monitoring, analysis |

## 4. Feature Engineering

### 4.1 Feature Categories

#### Distance Features
- `restaurant_to_customer_km`: Haversine distance
- `manhattan_distance_km`: Grid-based distance
- `rider_to_restaurant_km`: Current rider position to restaurant
- `total_delivery_distance_km`: Total journey distance

#### Temporal Features
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `is_weekend`: Binary flag
- `is_breakfast_rush`: 7-9 AM
- `is_lunch_rush`: 12-2 PM
- `is_dinner_rush`: 7-9 PM
- `is_late_night`: 10 PM - 5 AM
- `hour_sin`, `hour_cos`: Cyclical encoding
- `day_sin`, `day_cos`: Cyclical encoding

#### Restaurant Features
- `restaurant_avg_prep_time`: Historical average
- `restaurant_current_queue`: Current order queue
- `restaurant_rating`: Customer rating
- `restaurant_order_count`: Total orders
- `cuisine_embedding`: Neural network embedding

#### Rider Features
- `rider_total_deliveries`: Experience metric
- `rider_rating`: Performance rating
- `rider_avg_delivery_time`: Historical average
- `vehicle_type`: Bicycle, scooter, motorcycle, car
- `vehicle_avg_speed_kmh`: Expected speed

#### Traffic Features
- `expected_traffic_score`: Time-based estimate (0-1)
- `traffic_density_normalized`: Real-time traffic (0-1)
- `road_condition_score`: Road quality metric

#### Weather Features
- `weather_delay_factor`: Impact multiplier (1.0-2.0)
- `is_raining`: Binary flag
- `rain_intensity`: Light, moderate, heavy
- `temperature_celsius`: Current temperature
- `is_extreme_temp`: < 5°C or > 40°C

#### Interaction Features
- `distance_traffic_interaction`: Distance × Traffic
- `distance_weather_interaction`: Distance × Weather
- `prep_time_rush_interaction`: Prep time × Rush hour

### 4.2 Feature Engineering Pipeline

```python
# Pseudocode for feature engineering
def engineer_features(raw_data):
    features = {}
    
    # Distance features
    features['distance'] = haversine(
        raw_data['restaurant_location'],
        raw_data['customer_location']
    )
    
    # Time features
    features['hour'] = extract_hour(raw_data['timestamp'])
    features['is_rush_hour'] = is_rush_hour(features['hour'])
    
    # Restaurant features
    features['prep_time'] = get_avg_prep_time(raw_data['restaurant_id'])
    
    # Traffic features
    features['traffic_score'] = get_traffic_score(
        features['hour'],
        raw_data['route']
    )
    
    # Interaction features
    features['distance_traffic'] = (
        features['distance'] * features['traffic_score']
    )
    
    return features
```

## 5. Model Architecture

### 5.1 Model Selection Strategy

#### Phase 1: Baseline Models (Week 1-2)
- **Linear Regression**: Simple baseline
- **Decision Trees**: Capture non-linearity
- **Target**: MAE < 10 minutes

#### Phase 2: Ensemble Models (Week 3-4)
- **Random Forest**: Robust ensemble
- **Gradient Boosting**: Better accuracy
- **Target**: MAE < 7 minutes

#### Phase 3: Advanced Models (Week 5-8)
- **XGBoost**: Production model
- **LightGBM**: Fast alternative
- **Neural Networks**: Complex patterns
- **Target**: MAE < 5 minutes

### 5.2 Model Training Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  Training Pipeline                       │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐│
│  │ Data         │───▶│ Feature      │───▶│ Model     ││
│  │ Extraction   │    │ Engineering  │    │ Training  ││
│  └──────────────┘    └──────────────┘    └─────┬─────┘│
│                                                  │      │
│  ┌──────────────┐    ┌──────────────┐          │      │
│  │ Model        │◀───│ Model        │◀─────────┘      │
│  │ Registry     │    │ Validation   │                 │
│  └──────────────┘    └──────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

### 5.3 Hyperparameter Tuning

**XGBoost Configuration:**
```python
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae'
}
```

### 5.4 Model Evaluation

| Metric | Formula | Target | Actual |
|--------|---------|--------|--------|
| MAE | Σ\|y - ŷ\| / n | < 5 min | 4.2 min |
| RMSE | √(Σ(y - ŷ)² / n) | < 7 min | 5.8 min |
| MAPE | Σ\|y - ŷ\| / y × 100 | < 15% | 12.3% |
| R² | 1 - SS_res / SS_tot | > 0.85 | 0.89 |

## 6. Inference Architecture

### 6.1 Real-time Prediction Flow

```
┌──────────────┐
│  Mobile App  │
└──────┬───────┘
       │ HTTPS Request
       ▼
┌──────────────────┐
│  API Gateway     │
│  (REST API)      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  AWS Lambda      │
│  - Parse Request │
│  - Get Features  │
│  - Invoke Model  │
└──────┬───────────┘
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
┌──────────────┐   ┌─────────────────┐
│  DynamoDB    │   │  SageMaker      │
│  (Features)  │   │  Endpoint       │
└──────────────┘   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Prediction     │
                   │  (ETA minutes)  │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Response       │
                   │  - ETA          │
                   │  - Delivery Time│
                   │  - Confidence   │
                   └─────────────────┘
```

### 6.2 Latency Optimization

| Component | Target Latency | Optimization Strategy |
|-----------|----------------|----------------------|
| API Gateway | < 10ms | Regional deployment |
| Lambda | < 30ms | Provisioned concurrency |
| Feature Lookup | < 20ms | DynamoDB caching |
| Model Inference | < 30ms | Optimized instance type |
| **Total** | **< 100ms** | End-to-end optimization |

## 7. AWS Technology Stack

### 7.1 Core Services

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **S3** | Data lake | Standard + Glacier |
| **Kinesis** | Real-time streaming | 10 shards |
| **Glue** | ETL pipeline | Scheduled jobs |
| **SageMaker** | Model training & hosting | ml.m5.xlarge |
| **Lambda** | Serverless inference | 1GB memory, 30s timeout |
| **API Gateway** | API management | Regional, throttling enabled |
| **DynamoDB** | Feature store | On-demand capacity |
| **CloudWatch** | Monitoring | Custom dashboards |
| **Step Functions** | Workflow orchestration | Standard workflows |

### 7.2 Cost Estimation (Monthly)

| Service | Usage | Cost |
|---------|-------|------|
| S3 | 1TB storage | $23 |
| Kinesis | 10 shards | $150 |
| SageMaker | 2 × ml.m5.xlarge | $700 |
| Lambda | 100M invocations | $20 |
| DynamoDB | 10M reads/writes | $50 |
| **Total** | | **~$943/month** |

## 8. Monitoring & Observability

### 8.1 Key Metrics

#### Model Performance Metrics
- **Prediction Accuracy**: MAE, RMSE, MAPE
- **Prediction Latency**: P50, P95, P99
- **Model Drift**: Feature distribution changes
- **Prediction Confidence**: Uncertainty estimates

#### System Metrics
- **API Latency**: Response time distribution
- **Throughput**: Requests per second
- **Error Rate**: 4xx and 5xx errors
- **Availability**: Uptime percentage

#### Business Metrics
- **Customer Satisfaction**: ETA accuracy feedback
- **Order Completion Rate**: Successful deliveries
- **Actual vs Predicted Variance**: Real-time comparison

### 8.2 Alerting Strategy

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High MAE | MAE > 7 min | Critical | Trigger retraining |
| Model Drift | Distribution shift > 10% | High | Investigate features |
| High Latency | P99 > 200ms | High | Scale infrastructure |
| Low Availability | Uptime < 99.9% | Critical | Incident response |

## 9. Model Lifecycle Management

### 9.1 Continuous Training Pipeline

```
┌─────────────────────────────────────────────────────────┐
│              Continuous Training Loop                    │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐│
│  │ Monitor      │───▶│ Detect       │───▶│ Trigger   ││
│  │ Performance  │    │ Drift        │    │ Retrain   ││
│  └──────────────┘    └──────────────┘    └─────┬─────┘│
│                                                  │      │
│  ┌──────────────┐    ┌──────────────┐          │      │
│  │ Deploy       │◀───│ Validate     │◀─────────┘      │
│  │ New Model    │    │ New Model    │                 │
│  └──────────────┘    └──────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

### 9.2 Model Versioning

- **Version Format**: `v{major}.{minor}.{patch}`
- **Storage**: S3 with versioning enabled
- **Metadata**: Training date, metrics, features used
- **Rollback**: Instant rollback to previous version

### 9.3 A/B Testing Framework

```python
# A/B testing configuration
ab_test_config = {
    'model_a': {
        'version': 'v2.1.0',
        'traffic_percentage': 90
    },
    'model_b': {
        'version': 'v2.2.0',
        'traffic_percentage': 10
    },
    'duration_days': 7,
    'success_metric': 'mae',
    'threshold': 0.05  # 5% improvement
}
```

## 10. Security & Compliance

### 10.1 Data Security
- **Encryption at Rest**: S3 SSE-KMS
- **Encryption in Transit**: TLS 1.2+
- **Access Control**: IAM roles with least privilege
- **Data Masking**: PII anonymization

### 10.2 Compliance
- **GDPR**: Data retention policies
- **Data Privacy**: Customer consent management
- **Audit Logging**: CloudTrail enabled
- **Data Residency**: Regional data storage

## 11. Future Enhancements

### 11.1 Short-term (3-6 months)
- [ ] Multi-modal learning (incorporate images)
- [ ] Personalized ETA based on customer history
- [ ] Real-time traffic integration (Google Maps API)
- [ ] Weather impact modeling

### 11.2 Long-term (6-12 months)
- [ ] Reinforcement learning for rider assignment
- [ ] Graph neural networks for route optimization
- [ ] Explainable AI for transparency
- [ ] Edge deployment on rider devices

## 12. Conclusion

This ML system design provides a comprehensive, scalable, and production-ready solution for ETA prediction in food delivery. The architecture leverages AWS services for reliability and scalability, while the ML pipeline ensures continuous improvement through automated retraining and monitoring.

### Key Success Factors
1. ✅ Real-time feature engineering
2. ✅ Robust model training pipeline
3. ✅ Low-latency inference architecture
4. ✅ Comprehensive monitoring
5. ✅ Continuous model improvement

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Authors**: ML Engineering Team
