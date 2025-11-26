import React, { useState } from 'react';
import './LiveDemo.css';
import ETAPredictionForm from './ETAPredictionForm';
import PredictionResult from './PredictionResult';

const LiveDemo = ({ onPredict, predictionResult, isLoading }) => {
    return (
        <div className="live-demo">
            <div className="demo-header">
                <h2 className="demo-title">🚀 Live ETA Prediction Demo</h2>
                <p className="demo-description">
                    Enter order details to get real-time delivery time predictions powered by our ML model
                </p>
            </div>

            <div className="demo-grid">
                <div className="demo-section">
                    <ETAPredictionForm onSubmit={onPredict} isLoading={isLoading} />
                </div>

                <div className="demo-section">
                    <PredictionResult result={predictionResult} isLoading={isLoading} />
                </div>
            </div>

            {/* How it Works Section */}
            <div className="how-it-works">
                <h3 className="section-title">How It Works</h3>
                <div className="steps-grid">
                    <div className="step-card">
                        <div className="step-number">1</div>
                        <div className="step-icon">📍</div>
                        <h4 className="step-title">Input Data</h4>
                        <p className="step-description">
                            Enter restaurant and customer locations, along with order details
                        </p>
                    </div>

                    <div className="step-card">
                        <div className="step-number">2</div>
                        <div className="step-icon">⚙️</div>
                        <h4 className="step-title">Feature Engineering</h4>
                        <p className="step-description">
                            Calculate distance (Haversine), extract time features, assess traffic conditions
                        </p>
                    </div>

                    <div className="step-card">
                        <div className="step-number">3</div>
                        <div className="step-icon">🤖</div>
                        <h4 className="step-title">ML Prediction</h4>
                        <p className="step-description">
                            XGBoost model processes features and predicts delivery time
                        </p>
                    </div>

                    <div className="step-card">
                        <div className="step-number">4</div>
                        <div className="step-icon">⏱️</div>
                        <h4 className="step-title">Real-time ETA</h4>
                        <p className="step-description">
                            Get accurate delivery time with confidence intervals
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LiveDemo;
