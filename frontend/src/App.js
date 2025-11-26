import React, { useState } from 'react';
import './App.css';
import Header from './components/Header';
import SystemArchitecture from './components/SystemArchitecture';
import FeatureExplorer from './components/FeatureExplorer';
import ModelPerformance from './components/ModelPerformance';
import LiveDemo from './components/LiveDemo';
import Footer from './components/Footer';

function App() {
    const [activeTab, setActiveTab] = useState('demo');
    const [predictionResult, setPredictionResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    console.log("App Rendered. Active Tab:", activeTab);

    const handlePrediction = async (formData) => {
        setIsLoading(true);
        setPredictionResult(null);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    restaurant_lat: parseFloat(formData.restaurantLat),
                    restaurant_lon: parseFloat(formData.restaurantLon),
                    customer_lat: parseFloat(formData.customerLat),
                    customer_lon: parseFloat(formData.customerLon),
                    restaurant_name: formData.restaurantName,
                    prep_time: parseInt(formData.prepTime),
                    weather_condition: formData.weatherCondition,
                    order_time: formData.orderTime ? new Date().toISOString().split('T')[0] + ' ' + formData.orderTime + ':00' : null
                }),
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            const result = await response.json();
            setPredictionResult(result);
        } catch (error) {
            console.error("Prediction failed:", error);
            alert("Failed to get prediction. Please check the console for details.");
        } finally {
            setIsLoading(false);
        }
    };

    const tabs = [
        { id: 'demo', label: 'Live Demo', icon: '🚀' },
        { id: 'architecture', label: 'System Architecture', icon: '🏗️' },
        { id: 'features', label: 'Feature Engineering', icon: '⚙️' },
        { id: 'performance', label: 'Model Performance', icon: '📊' }
    ];

    return (
        <div className="App">
            <Header />

            <main className="main-content">
                {/* Navigation Tabs */}
                <div className="container">
                    <div className="tabs-container">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                                onClick={() => {
                                    console.log("Switching to tab:", tab.id);
                                    setActiveTab(tab.id);
                                }}
                            >
                                <span className="tab-icon">{tab.icon}</span>
                                <span className="tab-label">{tab.label}</span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Tab Content */}
                <div className="tab-content">
                    {activeTab === 'demo' && (
                        <div className="container" key="demo">
                            <LiveDemo
                                onPredict={handlePrediction}
                                predictionResult={predictionResult}
                                isLoading={isLoading}
                            />
                        </div>
                    )}

                    {activeTab === 'architecture' && (
                        <div className="container" key="architecture">
                            <SystemArchitecture />
                        </div>
                    )}

                    {activeTab === 'features' && (
                        <div className="container" key="features">
                            <FeatureExplorer />
                        </div>
                    )}

                    {activeTab === 'performance' && (
                        <div className="container" key="performance">
                            <ModelPerformance />
                        </div>
                    )}
                </div>
            </main>

            <Footer />
        </div>
    );
}

export default App;
