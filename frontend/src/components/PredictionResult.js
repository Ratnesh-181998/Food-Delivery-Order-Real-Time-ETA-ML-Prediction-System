import React from 'react';
import './PredictionResult.css';

const PredictionResult = ({ result, isLoading }) => {
    if (isLoading) {
        return (
            <div className="prediction-result-container card">
                <div className="loading-state">
                    <div className="spinner"></div>
                    <p className="loading-text">Calculating ETA...</p>
                    <p className="loading-subtext">Processing features and running ML model</p>
                </div>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="prediction-result-container card">
                <div className="empty-state">
                    <div className="empty-icon">📊</div>
                    <h3 className="empty-title">No Prediction Yet</h3>
                    <p className="empty-text">
                        Fill out the form and click "Predict ETA" to see results
                    </p>
                </div>
            </div>
        );
    }

    const getTrafficColor = (condition) => {
        switch (condition) {
            case 'low': return 'success';
            case 'medium': return 'warning';
            case 'high': return 'error';
            default: return 'info';
        }
    };

    const getTrafficIcon = (condition) => {
        switch (condition) {
            case 'low': return '🟢';
            case 'medium': return '🟡';
            case 'high': return '🔴';
            default: return '⚪';
        }
    };

    return (
        <div className="prediction-result-container card animate-fadeIn">
            <h3 className="result-title">🎯 Prediction Results</h3>

            {/* Main ETA Display */}
            <div className="eta-display">
                <div className="eta-main">
                    <div className="eta-label">Estimated Delivery Time</div>
                    <div className="eta-value">{result.eta_minutes} min</div>
                    <div className="eta-time">Arriving by {result.estimated_delivery_time}</div>
                </div>
                <div className="confidence-badge">
                    <div className="confidence-label">Confidence</div>
                    <div className="confidence-value">{(result.confidence_score * 100).toFixed(0)}%</div>
                </div>
            </div>

            {/* Breakdown */}
            <div className="breakdown-section">
                <h4 className="breakdown-title">📋 Breakdown</h4>
                <div className="breakdown-grid">
                    <div className="breakdown-item">
                        <div className="breakdown-icon">🍳</div>
                        <div className="breakdown-content">
                            <div className="breakdown-label">Prep Time</div>
                            <div className="breakdown-value">{result.prep_time_minutes} min</div>
                        </div>
                    </div>

                    <div className="breakdown-item">
                        <div className="breakdown-icon">📍</div>
                        <div className="breakdown-content">
                            <div className="breakdown-label">Distance</div>
                            <div className="breakdown-value">{result.distance_km} km</div>
                        </div>
                    </div>

                    <div className="breakdown-item">
                        <div className="breakdown-icon">{getTrafficIcon(result.traffic_condition)}</div>
                        <div className="breakdown-content">
                            <div className="breakdown-label">Traffic</div>
                            <div className="breakdown-value">
                                <span className={`badge badge-${getTrafficColor(result.traffic_condition)}`}>
                                    {result.traffic_condition}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="breakdown-item">
                        <div className="breakdown-icon">🏍️</div>
                        <div className="breakdown-content">
                            <div className="breakdown-label">Delivery</div>
                            <div className="breakdown-value">
                                {(result.eta_minutes - result.prep_time_minutes).toFixed(0)} min
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Features Used */}
            <div className="features-section">
                <h4 className="features-title">⚙️ Features Used</h4>
                <div className="features-list">
                    {Object.entries(result.features_used).map(([key, value]) => (
                        <div key={key} className="feature-item">
                            <span className="feature-key">{key.replace(/_/g, ' ')}</span>
                            <span className="feature-value">
                                {typeof value === 'boolean' ? (value ? '✓' : '✗') : value}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Confidence Interval */}
            <div className="confidence-section">
                <div className="confidence-bar-container">
                    <div className="confidence-bar">
                        <div
                            className="confidence-fill"
                            style={{ width: `${result.confidence_score * 100}%` }}
                        ></div>
                    </div>
                    <div className="confidence-labels">
                        <span>Low</span>
                        <span>Medium</span>
                        <span>High</span>
                    </div>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="action-buttons">
                <button className="btn btn-secondary">
                    <span>📊</span>
                    <span>View Details</span>
                </button>
                <button className="btn btn-primary">
                    <span>🔄</span>
                    <span>New Prediction</span>
                </button>
            </div>
        </div>
    );
};

export default PredictionResult;
