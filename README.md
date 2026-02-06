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


---


<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=24,20,12,6&height=3" width="100%">


## 📜 **License**

![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge&logo=opensourceinitiative&logoColor=white)

**Licensed under the MIT License** - Feel free to fork and build upon this innovation! 🚀

---

# 📞 **CONTACT & NETWORKING** 📞


### 💼 Professional Networks

[![LinkedIn](https://img.shields.io/badge/💼_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ratneshkumar1998/)
[![GitHub](https://img.shields.io/badge/🐙_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ratnesh-181998)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/RatneshS16497)
[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-FF6B6B?style=for-the-badge&logo=google-chrome&logoColor=white)](https://share.streamlit.io/user/ratnesh-181998)
[![Email](https://img.shields.io/badge/✉️_Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rattudacsit2021gate@gmail.com)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@rattudacsit2021gate)
[![Stack Overflow](https://img.shields.io/badge/Stack_Overflow-F58025?style=for-the-badge&logo=stack-overflow&logoColor=white)](https://stackoverflow.com/users/32068937/ratnesh-kumar)

### 🚀 AI/ML & Data Science
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io/user/ratnesh-181998)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/RattuDa98)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/rattuda)

### 💻 Competitive Programming (Including all coding plateform's 5000+ Problems/Questions solved )
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/u/Ratnesh_1998/)
[![HackerRank](https://img.shields.io/badge/HackerRank-00EA64?style=for-the-badge&logo=hackerrank&logoColor=black)](https://www.hackerrank.com/profile/rattudacsit20211)
[![CodeChef](https://img.shields.io/badge/CodeChef-5B4638?style=for-the-badge&logo=codechef&logoColor=white)](https://www.codechef.com/users/ratnesh_181998)
[![Codeforces](https://img.shields.io/badge/Codeforces-1F8ACB?style=for-the-badge&logo=codeforces&logoColor=white)](https://codeforces.com/profile/Ratnesh_181998)
[![GeeksforGeeks](https://img.shields.io/badge/GeeksforGeeks-2F8D46?style=for-the-badge&logo=geeksforgeeks&logoColor=white)](https://www.geeksforgeeks.org/profile/ratnesh1998)
[![HackerEarth](https://img.shields.io/badge/HackerEarth-323754?style=for-the-badge&logo=hackerearth&logoColor=white)](https://www.hackerearth.com/@ratnesh138/)
[![InterviewBit](https://img.shields.io/badge/InterviewBit-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://www.interviewbit.com/profile/rattudacsit2021gate_d9a25bc44230/)


---

## 📊 **GitHub Stats & Metrics** 📊



![Profile Views](https://komarev.com/ghpvc/?username=Ratnesh-181998&color=blueviolet&style=for-the-badge&label=PROFILE+VIEWS)




<img 
  src="https://streak-stats.demolab.com?user=Ratnesh-181998&theme=radical&hide_border=true&background=0D1117&stroke=4ECDC4&ring=F38181&fire=FF6B6B&currStreakLabel=4ECDC4"
  alt="GitHub Streak Stats"
width="48%"/>





<img src="https://github-readme-activity-graph.vercel.app/graph?username=Ratnesh-181998&theme=react-dark&hide_border=true&bg_color=0D1117&color=4ECDC4&line=F38181&point=FF6B6B" width="48%" />

---

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=4ECDC4&center=true&vCenter=true&width=600&lines=Ratnesh+Kumar+Singh;Data+Scientist+%7C+AI%2FML+Engineer;4%2B+Years+Building+Production+AI+Systems" alt="Typing SVG" />

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=F38181&center=true&vCenter=true&width=600&lines=Built+with+passion+for+the+AI+Community+🚀;Innovating+the+Future+of+AI+%26+ML;MLOps+%7C+LLMOps+%7C+AIOps+%7C+GenAI+%7C+AgenticAI+Excellence" alt="Footer Typing SVG" />


<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%">

