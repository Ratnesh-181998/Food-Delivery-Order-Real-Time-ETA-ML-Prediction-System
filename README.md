# 🛵 Zomato ETA Prediction System

![Zomato ETA Banner](project_showcase_ui/zomato_eta_github_banner.png)

> **Real-time Food Delivery Time Prediction using Machine Learning & AWS**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0%2B-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![AWS](https://img.shields.io/badge/AWS-Cloud-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-FLAT?style=for-the-badge&color=orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

## 📋 Overview

The **Zomato ETA Prediction System** is an end-to-end machine learning application designed to estimate food delivery times with high precision. By leveraging historical data, real-time traffic metrics, and weather conditions, this system provides accurate Estimated Time of Arrival (ETA) predictions to enhance customer satisfaction and logistics efficiency.

This project demonstrates a complete MLOps pipeline, from data ingestion and feature engineering to model training and deployment via a modern web interface.

## ✨ Key Features

*   **🚀 Real-Time Predictions:** Instant ETA calculation based on live inputs.
*   **🧠 Advanced ML Model:** Powered by **XGBoost** for superior tabular data performance (R² Score: 0.96).
*   **🌦️ Dynamic Factors:** Accounts for weather conditions (Rain, Fog, Clear) and traffic density.
*   **🏗️ Scalable Architecture:** Designed with AWS services (S3, SageMaker, Lambda) in mind.
*   **💻 Interactive UI:** A React-based dashboard for live demos, system architecture visualization, and performance monitoring.
*   **📊 Feature Engineering:** Custom transformers for Haversine distance, time-based features, and interaction terms.

## 🛠️ Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Frontend** | React.js, CSS3, Modern UI/UX Design |
| **Backend** | FastAPI (Python), Uvicorn |
| **Machine Learning** | XGBoost, Scikit-Learn, Pandas, NumPy |
| **Data Processing** | Joblib, Custom Feature Pipelines |
| **Deployment** | Docker (ready), AWS (Simulated Architecture) |

## 🏗️ System Architecture

The system follows a modern microservices-oriented architecture:

1.  **Data Ingestion:** Simulating real-time order streams.
2.  **Preprocessing:** Cleaning and transforming raw data into model-ready features.
3.  **Inference Engine:** A trained XGBoost regressor served via FastAPI.
4.  **Frontend:** A responsive web application consuming the prediction API.

## 🚀 Quick Start

### Prerequisites
*   Python 3.8+
*   Node.js 14+
*   npm

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Ratnesh-181998/zomato-eta-prediction.git
    cd zomato-eta-prediction
    ```

2.  **Backend Setup**
    ```bash
    cd backend
    pip install -r requirements.txt
    
    # Generate dummy training data (optional)
    python generate_training_data.py
    
    # Train the model
    python train_model.py
    ```

3.  **Frontend Setup**
    ```bash
    cd ../frontend
    npm install
    npm run build
    ```

4.  **Run the Application**
    Return to the `backend` directory and start the unified server:
    ```bash
    cd ../backend
    python main.py
    ```
    
    Access the application at: **http://localhost:8001**

## 📊 Model Performance

The model was trained on a synthetic dataset mimicking real-world delivery scenarios.

*   **Test MAE:** ~8.02 minutes
*   **Test RMSE:** ~10.45 minutes
*   **R² Score:** 0.9612

**Top Features:**
1.  `restaurant_to_customer_km` (Distance)
2.  `expected_traffic_score` (Traffic Density)
3.  `weather_delay_factor` (Weather Impact)

## 📸 Screenshots

### Live Demo
<img width="973" height="1476" alt="image" src="https://github.com/user-attachments/assets/e786ebc6-b625-4a09-93ad-52ab25dedfb5" />

### System Architecture
<img width="977" height="1473" alt="image" src="https://github.com/user-attachments/assets/afd5dc33-7522-4f05-9d52-63cadf0e2c3d" />
<img width="938" height="1167" alt="image" src="https://github.com/user-attachments/assets/7ec967be-c1be-49d0-898c-b7d336970a87" />

## 📊 Model Performance
Comprehensive evaluation of ML models and system performance metrics
<img width="980" height="1451" alt="image" src="https://github.com/user-attachments/assets/4572068d-23ac-4b26-91d0-4823a61c4341" />
<img width="945" height="1432" alt="image" src="https://github.com/user-attachments/assets/3904bbf3-15eb-49a4-b308-24b8c40dee17" />


## 🤝 Contact

**Ratnesh Kumar**

*   **GitHub:** [Ratnesh-181998](https://github.com/Ratnesh-181998)
*   **LinkedIn:** [Ratnesh Kumar](https://www.linkedin.com/in/ratneshkumar1998/)

---

*Made with ❤️ by Ratnesh Kumar*
