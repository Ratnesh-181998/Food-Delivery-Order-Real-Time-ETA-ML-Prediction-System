# Class Notes Reference: Zomato ETA Prediction ML System

> **Source**: Handwritten lecture notes on ML System Design for Zomato's ETA Prediction
> **Date**: November 2024

---

## 📱 Overview: Zomato ETA Prediction System

### System Flow
```
Zomato App → Real-time Prediction (↑↓) → Deployment
```

**Key Focus**: Data inception → Deployment

---

## 1️⃣ Requirements Analysis

### Functional Requirements

1. **Accurate ETA Predictions**
   - Real-time prediction (optional)

2. **Dynamic Updates**
   - Travel time
   - Restaurant prep time
   - Order complexity
   - Rider availability

3. **Restaurant & Rider Integration**

### Non-Functional Requirements

1. **Scalability** → Peak order processing
2. **Low Latency**
3. **High Availability / Reliability**
4. **Data Security & Privacy**

---

## 2️⃣ Data Sources & Structures

### 1. Order Data
- Order ID
- Restaurant ID
- Payment info
- Delivery address
- Special instructions

**Type**: Tabular, Structured

### 2. Restaurant Data
- Restaurant ID
- Name
- Address / Location (lat/long)
- Prep time (Chinese → Mughlai →)
- Operational hours
- Rating

**Note**: Different cuisines have different prep times

### 3. Rider Data
[Bracket notation in notes]

### 4. Customer Data
- Multiple locations
- **Framework**: Home

### 5. Traffic Data
- Road/NW data
- Speed, congestion (Google Maps API)
- Historic traffic pattern
- Realtime conditions

---

## 3️⃣ Storage Architecture

### Storage Solutions

| Data Type | Storage Solution | Purpose |
|-----------|-----------------|---------|
| Raw Data Lake | **Amazon S3** | General storage |
| Structured Data | **AWS Redshift** | Data warehouse |
| Frequently Accessed | **DynamoDB** | Fast queries |
| Metadata Store | **AWS Glue Data Catalog** | Data catalog |

---

## 4️⃣ Data Processing & Feature Engineering

### ETL Pipeline (Extract, Transform, Load)

**Tool**: AWS Glue (Information, Athena, DataBrew, DMS)

**Components**:
- **Bonus**: S3 + Lambda
- **AWS Glue** for:
  - Data cleaning ✓
  - Data Transformation ✓
  - Feature Engineering

### Feature Engineering Techniques

#### Distance Calculation
- **Haversine Formula** (accounts for Earth's curvature)
- Used for: Restaurant features, Rider-specific features

#### Time-Based Features
- Restaurant features
- Rider-specific features

#### Feature Encoding

1. **One-hot encoding**
2. **Target encoding**
3. **Ordinal encoding**
4. **Embeddings (Restaurant encoding)**
   - Reference: https://blog.zomato.com/food-preparation-time

---

## 5️⃣ Feature Scaling

**Categories**: Distance, Time, Order

### Encoding Methods

**Lat/Longitude (Encoding?)** → Geohashing

**Examples**:
- Amazon → 5 precision (ETA ↑↓ same)
- Swiggy → 9 precision

### POI (Points of Interest)
**Distance** → Landmarks

### Order Instructions
- TF-IDF
- Word2Vec, GloVe

---

## 6️⃣ Model Selection & Training Phase

### AWS Service
**EC2 (Virtual Machines)** → Training

### Model Progression

1. **Linear Regression** (Baseline)
2. **XGBoost, LightGBM** (Advanced)
3. **Neural Networks** (Deep Learning)

### Deployment
**SageMaker** (Training, Inference, Endpoint generation)
- Deployed ↓

---

## 7️⃣ Real-time Prediction Architecture

### Expose Endpoint as API

**AWS Lambda Function** handles:
1. Preprocess incoming data
2. (Includes transform)
3. Invoke SageMaker endpoint
4. Prediction
5. Post-process

---

## 8️⃣ Advanced Concepts

### 1. User Experience Enhancement

**Range of possible arrival time (calculated)**
- **MEAN**
- **+ Confidence intervals**

### 2. ETA vs Arrival (Actual)

**Visualization**:
```
ETA:     |-------|
Arrival: |-------|
```

**Mean of both groups are almost**:
- ↳ Significant
- ↳ Hypothesis testing

**Key Insight**: Retraining is needed
- ↳ Add more features

---

## 🔑 Key Takeaways

### Critical Components

1. **Data Pipeline**
   - Multiple data sources (Orders, Restaurants, Riders, Traffic, Weather)
   - AWS-based storage (S3, Redshift, DynamoDB)
   - ETL using AWS Glue

2. **Feature Engineering**
   - Haversine distance for geospatial calculations
   - Time-based features
   - Embeddings for high-cardinality categorical data
   - Geohashing for location encoding

3. **Model Development**
   - Start simple: Linear Regression
   - Progress to: XGBoost, LightGBM
   - Advanced: Neural Networks
   - Train on EC2, Deploy on SageMaker

4. **Real-time Inference**
   - AWS Lambda for API handling
   - SageMaker endpoints for predictions
   - Low latency requirements

5. **Continuous Improvement**
   - Monitor ETA vs Actual arrival times
   - Hypothesis testing for model drift
   - Retrain when performance degrades
   - Add features as needed

---

## 📊 System Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                   ZOMATO ETA SYSTEM                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Data Sources                                           │
│  ├── Orders (Structured)                                │
│  ├── Restaurants (Structured + Embeddings)              │
│  ├── Riders (Real-time GPS)                             │
│  ├── Traffic (Google Maps API)                          │
│  └── Weather (External API)                             │
│                                                          │
│  Storage Layer                                          │
│  ├── S3 (Data Lake)                                     │
│  ├── Redshift (Data Warehouse)                          │
│  ├── DynamoDB (Real-time Features)                      │
│  └── Glue Data Catalog (Metadata)                       │
│                                                          │
│  Processing Layer                                       │
│  ├── AWS Glue (ETL)                                     │
│  ├── Feature Engineering                                │
│  │   ├── Haversine Distance                             │
│  │   ├── Time Features                                  │
│  │   ├── Embeddings                                     │
│  │   └── Geohashing                                     │
│  └── Feature Scaling                                    │
│                                                          │
│  Model Layer                                            │
│  ├── Training (EC2)                                     │
│  │   ├── Linear Regression                              │
│  │   ├── XGBoost / LightGBM                             │
│  │   └── Neural Networks                                │
│  └── Deployment (SageMaker)                             │
│                                                          │
│  Inference Layer                                        │
│  ├── AWS Lambda (API Handler)                           │
│  ├── SageMaker Endpoint                                 │
│  └── Real-time Prediction                               │
│                                                          │
│  Monitoring & Improvement                               │
│  ├── ETA vs Actual Comparison                           │
│  ├── Hypothesis Testing                                 │
│  └── Model Retraining                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Important Formulas & Concepts

### Haversine Distance Formula
```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points
    Accounts for Earth's curvature
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth's radius in km
    
    return c * r
```

### Geohashing Precision
- **Amazon**: 5 precision (broader area)
- **Swiggy**: 9 precision (more granular)

### Confidence Intervals
- Provide range of possible arrival times
- Calculate mean + confidence intervals
- Better UX than single point estimate

---

## 📚 External References

1. **Zomato Blog**: Food Preparation Time
   - https://blog.zomato.com/food-preparation-time

2. **Google Maps API**: Traffic data integration

3. **AWS Services**:
   - S3, Redshift, DynamoDB
   - Glue, Lambda, SageMaker
   - EC2

---

## 💡 Best Practices Highlighted

1. **Start Simple**: Begin with Linear Regression baseline
2. **Iterate**: Progress to more complex models (XGBoost → Neural Networks)
3. **Monitor**: Continuously compare ETA vs Actual
4. **Test**: Use hypothesis testing to detect model drift
5. **Improve**: Retrain and add features when needed
6. **Scale**: Use AWS services for scalability and reliability
7. **Optimize**: Use appropriate encoding (embeddings, geohashing)
8. **User-Centric**: Provide confidence intervals, not just point estimates

---

**Document Created**: November 26, 2024  
**Based on**: Handwritten class notes (5 pages)  
**Topic**: ML System Design - Zomato ETA Prediction
