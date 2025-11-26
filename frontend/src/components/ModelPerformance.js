import React from 'react';
import './ModelPerformance.css';

const ModelPerformance = () => {
    const modelComparison = [
        { model: 'Linear Regression', mae: 8.5, rmse: 11.2, r2: 0.65, training_time: '2 min' },
        { model: 'Random Forest', mae: 5.8, rmse: 7.9, r2: 0.82, training_time: '15 min' },
        { model: 'XGBoost', mae: 4.2, rmse: 5.8, r2: 0.89, training_time: '25 min' },
        { model: 'LightGBM', mae: 4.3, rmse: 5.9, r2: 0.88, training_time: '18 min' },
        { model: 'Neural Network', mae: 4.8, rmse: 6.5, r2: 0.86, training_time: '45 min' }
    ];

    const featureImportance = [
        { feature: 'restaurant_to_customer_km', importance: 0.28 },
        { feature: 'restaurant_avg_prep_time', importance: 0.22 },
        { feature: 'expected_traffic_score', importance: 0.18 },
        { feature: 'hour', importance: 0.12 },
        { feature: 'is_dinner_rush', importance: 0.08 },
        { feature: 'weather_delay_factor', importance: 0.06 },
        { feature: 'day_of_week', importance: 0.04 },
        { feature: 'rider_total_deliveries', importance: 0.02 }
    ];

    const performanceMetrics = [
        { metric: 'Mean Absolute Error (MAE)', value: '4.2 min', target: '< 5 min', status: 'success' },
        { metric: 'Root Mean Squared Error (RMSE)', value: '5.8 min', target: '< 7 min', status: 'success' },
        { metric: 'R² Score', value: '0.89', target: '> 0.85', status: 'success' },
        { metric: 'MAPE', value: '12.3%', target: '< 15%', status: 'success' },
        { metric: 'Prediction Latency (P99)', value: '87 ms', target: '< 100 ms', status: 'success' },
        { metric: 'Throughput', value: '1.2M/day', target: '> 1M/day', status: 'success' }
    ];

    return (
        <div className="model-performance">
            <div className="performance-header">
                <h2 className="performance-title">📊 Model Performance</h2>
                <p className="performance-description">
                    Comprehensive evaluation of ML models and system performance metrics
                </p>
            </div>

            {/* Performance Metrics Grid */}
            <div className="metrics-grid">
                {performanceMetrics.map((metric, idx) => (
                    <div key={idx} className="metric-card">
                        <div className="metric-header">
                            <span className={`status-indicator status-${metric.status}`}></span>
                            <span className="metric-name">{metric.metric}</span>
                        </div>
                        <div className="metric-value">{metric.value}</div>
                        <div className="metric-target">Target: {metric.target}</div>
                    </div>
                ))}
            </div>

            {/* Model Comparison Table */}
            <div className="comparison-section">
                <h3 className="section-title">🤖 Model Comparison</h3>
                <div className="table-container">
                    <table className="comparison-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>MAE (min)</th>
                                <th>RMSE (min)</th>
                                <th>R² Score</th>
                                <th>Training Time</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {modelComparison.map((model, idx) => (
                                <tr key={idx} className={model.model === 'XGBoost' ? 'best-model' : ''}>
                                    <td className="model-name">
                                        {model.model}
                                        {model.model === 'XGBoost' && <span className="best-badge">Best</span>}
                                    </td>
                                    <td>{model.mae}</td>
                                    <td>{model.rmse}</td>
                                    <td>{model.r2}</td>
                                    <td>{model.training_time}</td>
                                    <td>
                                        <span className={`badge ${model.mae < 5 ? 'badge-success' : 'badge-warning'}`}>
                                            {model.mae < 5 ? 'Excellent' : 'Good'}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Feature Importance */}
            <div className="importance-section">
                <h3 className="section-title">⭐ Feature Importance</h3>
                <div className="importance-chart">
                    {featureImportance.map((item, idx) => (
                        <div key={idx} className="importance-item">
                            <div className="importance-label">
                                <span className="importance-rank">#{idx + 1}</span>
                                <span className="importance-name">{item.feature.replace(/_/g, ' ')}</span>
                            </div>
                            <div className="importance-bar-container">
                                <div
                                    className="importance-bar"
                                    style={{ width: `${item.importance * 100}%` }}
                                >
                                    <span className="importance-value">{(item.importance * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Training Details */}
            <div className="training-details">
                <h3 className="section-title">🎯 XGBoost Model Details</h3>
                <div className="details-grid">
                    <div className="detail-card">
                        <div className="detail-label">Algorithm</div>
                        <div className="detail-value">XGBoost Regressor</div>
                    </div>
                    <div className="detail-card">
                        <div className="detail-label">Objective</div>
                        <div className="detail-value">reg:squarederror</div>
                    </div>
                    <div className="detail-card">
                        <div className="detail-label">N Estimators</div>
                        <div className="detail-value">1000</div>
                    </div>
                    <div className="detail-card">
                        <div className="detail-label">Learning Rate</div>
                        <div className="detail-value">0.01</div>
                    </div>
                    <div className="detail-card">
                        <div className="detail-label">Max Depth</div>
                        <div className="detail-value">7</div>
                    </div>
                    <div className="detail-card">
                        <div className="detail-label">Subsample</div>
                        <div className="detail-value">0.8</div>
                    </div>
                    <div className="detail-card">
                        <div className="detail-label">Training Samples</div>
                        <div className="detail-value">2.5M orders</div>
                    </div>
                    <div className="detail-card">
                        <div className="detail-label">Validation Split</div>
                        <div className="detail-value">20%</div>
                    </div>
                </div>
            </div>

            {/* Deployment Info */}
            <div className="deployment-info">
                <h3 className="section-title">🚀 Deployment Configuration</h3>
                <div className="deployment-grid">
                    <div className="deployment-card">
                        <div className="deployment-icon">☁️</div>
                        <div className="deployment-title">Cloud Platform</div>
                        <div className="deployment-value">AWS SageMaker</div>
                    </div>
                    <div className="deployment-card">
                        <div className="deployment-icon">💻</div>
                        <div className="deployment-title">Instance Type</div>
                        <div className="deployment-value">ml.m5.xlarge (×2)</div>
                    </div>
                    <div className="deployment-card">
                        <div className="deployment-icon">📈</div>
                        <div className="deployment-title">Auto Scaling</div>
                        <div className="deployment-value">2-10 instances</div>
                    </div>
                    <div className="deployment-card">
                        <div className="deployment-icon">🔄</div>
                        <div className="deployment-title">Update Frequency</div>
                        <div className="deployment-value">Weekly retraining</div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ModelPerformance;
