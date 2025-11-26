import React, { useState } from 'react';
import './ETAPredictionForm.css';

const ETAPredictionForm = ({ onSubmit, isLoading }) => {
    const [formData, setFormData] = useState({
        restaurantLat: '28.6139',
        restaurantLon: '77.2090',
        customerLat: '28.6289',
        customerLon: '77.2190',
        distance: '',
        restaurantName: 'Dominos Pizza',
        prepTime: '20',
        weatherCondition: 'clear',
        orderTime: new Date().toTimeString().slice(0, 5)
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit(formData);
    };

    const presetLocations = [
        { name: 'Connaught Place → Karol Bagh', restLat: '28.6315', restLon: '77.2167', custLat: '28.6519', custLon: '77.1909' },
        { name: 'Hauz Khas → Saket', restLat: '28.5494', restLon: '77.2001', custLat: '28.5244', custLon: '77.2066' },
        { name: 'Nehru Place → Lajpat Nagar', restLat: '28.5494', restLon: '77.2501', custLat: '28.5678', custLon: '77.2434' }
    ];

    const loadPreset = (preset) => {
        setFormData(prev => ({
            ...prev,
            restaurantLat: preset.restLat,
            restaurantLon: preset.restLon,
            customerLat: preset.custLat,
            customerLon: preset.custLon,
            distance: ''
        }));
    };

    return (
        <div className="prediction-form-container card">
            <h3 className="form-title">📝 Order Details</h3>

            {/* Preset Locations */}
            <div className="preset-section">
                <label className="preset-label">Quick Presets:</label>
                <div className="preset-buttons">
                    {presetLocations.map((preset, idx) => (
                        <button
                            key={idx}
                            type="button"
                            className="preset-btn"
                            onClick={() => loadPreset(preset)}
                        >
                            {preset.name}
                        </button>
                    ))}
                </div>
            </div>

            <form onSubmit={handleSubmit} className="prediction-form">
                {/* Restaurant Location */}
                <div className="form-section">
                    <h4 className="section-heading">🏪 Restaurant Location</h4>
                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label">Restaurant Name</label>
                            <input
                                type="text"
                                name="restaurantName"
                                value={formData.restaurantName}
                                onChange={handleChange}
                                className="form-input"
                                placeholder="e.g., Dominos Pizza"
                            />
                        </div>
                    </div>
                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label">Latitude</label>
                            <input
                                type="number"
                                step="0.0001"
                                name="restaurantLat"
                                value={formData.restaurantLat}
                                onChange={handleChange}
                                className="form-input"
                                required
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Longitude</label>
                            <input
                                type="number"
                                step="0.0001"
                                name="restaurantLon"
                                value={formData.restaurantLon}
                                onChange={handleChange}
                                className="form-input"
                                required
                            />
                        </div>
                    </div>
                </div>

                {/* Customer Location */}
                <div className="form-section">
                    <h4 className="section-heading">📍 Customer Location</h4>
                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label">Latitude</label>
                            <input
                                type="number"
                                step="0.0001"
                                name="customerLat"
                                value={formData.customerLat}
                                onChange={handleChange}
                                className="form-input"
                                required
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Longitude</label>
                            <input
                                type="number"
                                step="0.0001"
                                name="customerLon"
                                value={formData.customerLon}
                                onChange={handleChange}
                                className="form-input"
                                required
                            />
                        </div>
                    </div>
                </div>

                {/* Order Details */}
                <div className="form-section">
                    <h4 className="section-heading">⏰ Order Details</h4>
                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label">Prep Time (minutes)</label>
                            <input
                                type="number"
                                name="prepTime"
                                value={formData.prepTime}
                                onChange={handleChange}
                                className="form-input"
                                min="5"
                                max="60"
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Order Time</label>
                            <input
                                type="time"
                                name="orderTime"
                                value={formData.orderTime}
                                onChange={handleChange}
                                className="form-input"
                            />
                        </div>
                    </div>
                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label">Weather Condition</label>
                            <select
                                name="weatherCondition"
                                value={formData.weatherCondition}
                                onChange={handleChange}
                                className="form-select"
                            >
                                <option value="clear">☀️ Clear</option>
                                <option value="cloudy">☁️ Cloudy</option>
                                <option value="light_rain">🌧️ Light Rain</option>
                                <option value="heavy_rain">⛈️ Heavy Rain</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Distance (km) - Optional</label>
                            <input
                                type="number"
                                step="0.1"
                                name="distance"
                                value={formData.distance}
                                onChange={handleChange}
                                className="form-input"
                                placeholder="Auto-calculated"
                            />
                        </div>
                    </div>
                </div>

                {/* Submit Button */}
                <button
                    type="submit"
                    className="btn btn-primary submit-btn"
                    disabled={isLoading}
                >
                    {isLoading ? (
                        <>
                            <div className="spinner-small"></div>
                            <span>Predicting...</span>
                        </>
                    ) : (
                        <>
                            <span>🚀</span>
                            <span>Predict ETA</span>
                        </>
                    )}
                </button>
            </form>
        </div>
    );
};

export default ETAPredictionForm;
