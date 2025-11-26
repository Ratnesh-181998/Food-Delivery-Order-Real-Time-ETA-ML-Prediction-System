import React from 'react';
import './FeatureExplorer.css';

const FeatureExplorer = () => {
    const featureCategories = [
        {
            title: 'Distance Features',
            icon: '📍',
            color: '#3B82F6',
            features: [
                {
                    name: 'Haversine Distance',
                    formula: 'd = 2r × arcsin(√(sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)))',
                    description: 'Great circle distance accounting for Earth\'s curvature',
                    importance: 'High',
                    example: '5.2 km'
                },
                {
                    name: 'Manhattan Distance',
                    formula: 'd = |lat₂ - lat₁| × 111 + |lon₂ - lon₁| × 111 × cos(avg_lat)',
                    description: 'Grid-based distance for urban areas',
                    importance: 'Medium',
                    example: '6.8 km'
                }
            ]
        },
        {
            title: 'Temporal Features',
            icon: '⏰',
            color: '#10B981',
            features: [
                {
                    name: 'Hour of Day',
                    formula: 'hour = timestamp.hour (0-23)',
                    description: 'Cyclical encoding: sin(2π × hour/24), cos(2π × hour/24)',
                    importance: 'High',
                    example: '19 (7 PM)'
                },
                {
                    name: 'Day of Week',
                    formula: 'day = timestamp.weekday() (0-6)',
                    description: 'Monday=0, Sunday=6, with cyclical encoding',
                    importance: 'Medium',
                    example: '5 (Saturday)'
                },
                {
                    name: 'Rush Hour Indicators',
                    formula: 'is_rush = hour in [12-14, 19-21]',
                    description: 'Binary flags for breakfast, lunch, dinner rush',
                    importance: 'High',
                    example: 'is_dinner_rush = 1'
                }
            ]
        },
        {
            title: 'Restaurant Features',
            icon: '🏪',
            color: '#F59E0B',
            features: [
                {
                    name: 'Average Prep Time',
                    formula: 'avg_prep = mean(historical_prep_times)',
                    description: 'Restaurant-specific preparation time from history',
                    importance: 'High',
                    example: '22 minutes'
                },
                {
                    name: 'Cuisine Embeddings',
                    formula: 'embedding = neural_network(cuisine_type)',
                    description: 'Dense vector representation of cuisine categories',
                    importance: 'Medium',
                    example: '[0.23, -0.45, 0.67, ...]'
                },
                {
                    name: 'Restaurant Rating',
                    formula: 'rating = customer_ratings.mean()',
                    description: 'Average customer rating (1-5 stars)',
                    importance: 'Low',
                    example: '4.5 stars'
                }
            ]
        },
        {
            title: 'Traffic Features',
            icon: '🚦',
            color: '#EF4444',
            features: [
                {
                    name: 'Traffic Density Score',
                    formula: 'traffic_score = (current_density - min) / (max - min)',
                    description: 'Normalized traffic density from 0 (clear) to 1 (heavy)',
                    importance: 'High',
                    example: '0.75 (heavy traffic)'
                },
                {
                    name: 'Expected Traffic',
                    formula: 'expected = time_based_lookup[hour, day, location]',
                    description: 'Historical traffic patterns for time and location',
                    importance: 'High',
                    example: '0.85 (rush hour)'
                }
            ]
        },
        {
            title: 'Weather Features',
            icon: '🌤️',
            color: '#8B5CF6',
            features: [
                {
                    name: 'Weather Delay Factor',
                    formula: 'delay = weather_impact_map[condition]',
                    description: 'Multiplier based on weather: clear=1.0, rain=1.2, storm=2.0',
                    importance: 'Medium',
                    example: '1.2 (light rain)'
                },
                {
                    name: 'Temperature Impact',
                    formula: 'is_extreme = temp < 5°C or temp > 40°C',
                    description: 'Binary flag for extreme temperatures affecting delivery',
                    importance: 'Low',
                    example: '0 (normal temp)'
                }
            ]
        },
        {
            title: 'Interaction Features',
            icon: '🔗',
            color: '#EC4899',
            features: [
                {
                    name: 'Distance × Traffic',
                    formula: 'interaction = distance_km × traffic_score',
                    description: 'Combined impact of distance and traffic conditions',
                    importance: 'High',
                    example: '5.2 × 0.75 = 3.9'
                },
                {
                    name: 'Distance × Weather',
                    formula: 'interaction = distance_km × weather_delay_factor',
                    description: 'Combined impact of distance and weather',
                    importance: 'Medium',
                    example: '5.2 × 1.2 = 6.24'
                }
            ]
        }
    ];

    return (
        <div className="feature-explorer">
            <div className="explorer-header">
                <h2 className="explorer-title">⚙️ Feature Engineering</h2>
                <p className="explorer-description">
                    Comprehensive feature engineering pipeline transforming raw data into ML-ready features
                </p>
            </div>

            {/* Feature Categories */}
            <div className="categories-grid">
                {featureCategories.map((category, idx) => (
                    <div key={idx} className="category-card">
                        <div className="category-header" style={{ background: category.color }}>
                            <span className="category-icon">{category.icon}</span>
                            <h3 className="category-title">{category.title}</h3>
                        </div>

                        <div className="category-features">
                            {category.features.map((feature, fidx) => (
                                <div key={fidx} className="feature-detail">
                                    <div className="feature-header-row">
                                        <h4 className="feature-name">{feature.name}</h4>
                                        <span className={`importance-badge importance-${feature.importance.toLowerCase()}`}>
                                            {feature.importance}
                                        </span>
                                    </div>

                                    <div className="feature-formula">
                                        <code>{feature.formula}</code>
                                    </div>

                                    <p className="feature-description">{feature.description}</p>

                                    <div className="feature-example">
                                        <span className="example-label">Example:</span>
                                        <span className="example-value">{feature.example}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            {/* Feature Engineering Pipeline */}
            <div className="pipeline-section">
                <h3 className="section-title">🔄 Feature Engineering Pipeline</h3>
                <div className="pipeline-steps">
                    <div className="pipeline-step">
                        <div className="step-badge">1</div>
                        <div className="step-content">
                            <h4>Data Extraction</h4>
                            <p>Extract raw data from multiple sources (orders, restaurants, riders, external APIs)</p>
                        </div>
                    </div>

                    <div className="pipeline-arrow">↓</div>

                    <div className="pipeline-step">
                        <div className="step-badge">2</div>
                        <div className="step-content">
                            <h4>Feature Calculation</h4>
                            <p>Calculate distance (Haversine), extract time features, retrieve historical data</p>
                        </div>
                    </div>

                    <div className="pipeline-arrow">↓</div>

                    <div className="pipeline-step">
                        <div className="step-badge">3</div>
                        <div className="step-content">
                            <h4>Feature Encoding</h4>
                            <p>Apply one-hot encoding, embeddings, cyclical encoding for categorical variables</p>
                        </div>
                    </div>

                    <div className="pipeline-arrow">↓</div>

                    <div className="pipeline-step">
                        <div className="step-badge">4</div>
                        <div className="step-content">
                            <h4>Feature Scaling</h4>
                            <p>Normalize/standardize numerical features for model training</p>
                        </div>
                    </div>

                    <div className="pipeline-arrow">↓</div>

                    <div className="pipeline-step">
                        <div className="step-badge">5</div>
                        <div className="step-content">
                            <h4>Feature Store</h4>
                            <p>Store processed features in DynamoDB for real-time inference</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FeatureExplorer;
