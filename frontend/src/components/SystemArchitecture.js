import React from 'react';
import './SystemArchitecture.css';

const SystemArchitecture = () => {
    const architectureLayers = [
        {
            title: 'Data Sources',
            icon: '💾',
            components: [
                { name: 'Order Management System', desc: 'Order details, customer info, payment' },
                { name: 'Restaurant Database', desc: 'Location, prep times, ratings' },
                { name: 'Rider Tracking System', desc: 'GPS location, availability, performance' },
                { name: 'External APIs', desc: 'Traffic (Google Maps), Weather data' }
            ]
        },
        {
            title: 'Data Ingestion',
            icon: '📥',
            components: [
                { name: 'AWS Kinesis Streams', desc: 'Real-time data streaming' },
                { name: 'AWS Kinesis Firehose', desc: 'Streaming ETL to S3' },
                { name: 'API Gateway', desc: 'RESTful API endpoints' }
            ]
        },
        {
            title: 'Storage Layer',
            icon: '🗄️',
            components: [
                { name: 'Amazon S3', desc: 'Data lake for raw and processed data' },
                { name: 'AWS Redshift', desc: 'Data warehouse for analytics' },
                { name: 'DynamoDB', desc: 'Real-time feature store' },
                { name: 'AWS Glue Data Catalog', desc: 'Metadata management' }
            ]
        },
        {
            title: 'Processing & Feature Engineering',
            icon: '⚙️',
            components: [
                { name: 'AWS Glue', desc: 'ETL jobs for data transformation' },
                { name: 'Feature Engineering Pipeline', desc: 'Haversine distance, time features, embeddings' },
                { name: 'AWS Lambda', desc: 'Serverless data processing' }
            ]
        },
        {
            title: 'Model Training',
            icon: '🤖',
            components: [
                { name: 'AWS SageMaker', desc: 'Model training and hyperparameter tuning' },
                { name: 'EC2 Instances', desc: 'Distributed training for large datasets' },
                { name: 'Model Registry', desc: 'Version control and model artifacts' }
            ]
        },
        {
            title: 'Inference & Deployment',
            icon: '🚀',
            components: [
                { name: 'SageMaker Endpoints', desc: 'Real-time model serving' },
                { name: 'AWS Lambda Functions', desc: 'API request handling and preprocessing' },
                { name: 'Auto Scaling', desc: 'Dynamic scaling based on load' }
            ]
        },
        {
            title: 'Monitoring & Ops',
            icon: '📊',
            components: [
                { name: 'AWS CloudWatch', desc: 'Metrics, logs, and alarms' },
                { name: 'Model Monitoring', desc: 'Drift detection and performance tracking' },
                { name: 'A/B Testing Framework', desc: 'Continuous experimentation' }
            ]
        }
    ];

    const awsServices = [
        { name: 'S3', icon: '🪣', color: '#569A31' },
        { name: 'Kinesis', icon: '🌊', color: '#8C4FFF' },
        { name: 'Lambda', icon: 'λ', color: '#FF9900' },
        { name: 'SageMaker', icon: '🧠', color: '#FF9900' },
        { name: 'DynamoDB', icon: '⚡', color: '#4053D6' },
        { name: 'Glue', icon: '🔗', color: '#8C4FFF' },
        { name: 'CloudWatch', icon: '👁️', color: '#FF4F8B' },
        { name: 'EC2', icon: '💻', color: '#FF9900' }
    ];

    return (
        <div className="system-architecture">
            <div className="architecture-header">
                <h2 className="architecture-title">🏗️ System Architecture</h2>
                <p className="architecture-description">
                    End-to-end ML system design for real-time ETA prediction using AWS cloud services
                </p>
            </div>

            {/* AWS Services Grid */}
            <div className="aws-services-section">
                <h3 className="section-title">AWS Services Used</h3>
                <div className="aws-services-grid">
                    {awsServices.map((service, idx) => (
                        <div key={idx} className="aws-service-card" style={{ borderColor: service.color }}>
                            <div className="service-icon" style={{ background: service.color }}>
                                {service.icon}
                            </div>
                            <div className="service-name">{service.name}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Architecture Layers */}
            <div className="architecture-layers">
                {architectureLayers.map((layer, idx) => (
                    <div key={idx} className="layer-card">
                        <div className="layer-header">
                            <span className="layer-icon">{layer.icon}</span>
                            <h3 className="layer-title">{layer.title}</h3>
                            <span className="layer-number">{idx + 1}</span>
                        </div>
                        <div className="layer-components">
                            {layer.components.map((component, cidx) => (
                                <div key={cidx} className="component-item">
                                    <div className="component-name">{component.name}</div>
                                    <div className="component-desc">{component.desc}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            {/* Data Flow Diagram */}
            <div className="data-flow-section">
                <h3 className="section-title">📊 Data Flow</h3>
                <div className="flow-diagram">
                    <div className="flow-step">
                        <div className="flow-box">Data Sources</div>
                        <div className="flow-arrow">→</div>
                    </div>
                    <div className="flow-step">
                        <div className="flow-box">Kinesis Streams</div>
                        <div className="flow-arrow">→</div>
                    </div>
                    <div className="flow-step">
                        <div className="flow-box">S3 Data Lake</div>
                        <div className="flow-arrow">→</div>
                    </div>
                    <div className="flow-step">
                        <div className="flow-box">AWS Glue ETL</div>
                        <div className="flow-arrow">→</div>
                    </div>
                    <div className="flow-step">
                        <div className="flow-box">Feature Store</div>
                        <div className="flow-arrow">→</div>
                    </div>
                    <div className="flow-step">
                        <div className="flow-box">SageMaker</div>
                        <div className="flow-arrow">→</div>
                    </div>
                    <div className="flow-step">
                        <div className="flow-box">Prediction API</div>
                    </div>
                </div>
            </div>

            {/* Key Features */}
            <div className="key-features-section">
                <h3 className="section-title">✨ Key Features</h3>
                <div className="features-grid">
                    <div className="feature-card">
                        <div className="feature-icon">⚡</div>
                        <h4 className="feature-title">Real-time Processing</h4>
                        <p className="feature-desc">Sub-100ms latency for ETA predictions</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">📈</div>
                        <h4 className="feature-title">Scalable Architecture</h4>
                        <p className="feature-desc">Handle 1M+ predictions per day</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">🔄</div>
                        <h4 className="feature-title">Continuous Learning</h4>
                        <p className="feature-desc">Automated model retraining pipeline</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">🛡️</div>
                        <h4 className="feature-title">High Availability</h4>
                        <p className="feature-desc">99.99% uptime with auto-scaling</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SystemArchitecture;
