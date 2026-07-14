# 🚗 AutoValueAI

<p align="center">
  <strong>AI-Powered Used Car Price Prediction Platform</strong><br>
  An end-to-end Machine Learning application with an interactive Streamlit dashboard and FastAPI integration.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange.svg">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red.svg">
  <img src="https://img.shields.io/badge/FastAPI-REST_API-green.svg">
</p>

---

## 📖 Overview

**AutoValueAI** is a production-inspired Machine Learning application designed to estimate the selling price of used vehicles. The project combines data preprocessing, exploratory data analysis, machine learning model development, interactive visualizations, and REST API deployment into one complete solution.

Rather than focusing only on model training, this project demonstrates the complete Machine Learning workflow—from raw data to an interactive prediction platform.

To run it dashboard open browser and copy this adddress and paste it on the browser
http://127.0.0.1:8504
---

## 🎯 Key Features

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Feature Engineering
- 🤖 Multiple Machine Learning Models
  - Linear Regression
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
- 📈 Model Performance Evaluation
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- 📉 Interactive Plotly Visualizations
- 🚗 Real-Time Vehicle Price Prediction
- 🌐 Modern Streamlit Dashboard
- ⚡ FastAPI REST API
- 📱 User-Friendly Interface

---

# 🏗️ Project Workflow

```text
                Dataset
                   │
                   ▼
        Data Preprocessing
                   │
                   ▼
        Feature Engineering
                   │
                   ▼
         Model Development
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
Linear Regression Random Forest   SVR
        │          │          │
        └──────────┼──────────┘
                   ▼
         Model Evaluation
                   │
                   ▼
     Streamlit Dashboard + FastAPI
                   │
                   ▼
      Real-Time Price Prediction
```

---

# 🛠 Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly |
| Web Application | Streamlit |
| API Development | FastAPI, Pydantic |

---

# 📊 Dashboard Modules

### 🏠 Home
- Dataset Overview
- Vehicle Statistics
- Market Insights

### 📈 Analytics
- Interactive Charts
- Price Distribution
- Fuel Type Analysis
- Market Trends

### 🤖 Model Lab
- Train Multiple Models
- Compare Performance
- Hyperparameter Optimization
- Model Evaluation Metrics

### 🚗 Prediction
- User Input Form
- Real-Time Price Estimation
- Interactive Controls

### ⚡ API
- REST Endpoint
- JSON Requests
- JSON Responses

---

# 🤖 Machine Learning Models

| Model | Purpose |
|--------|----------|
| Linear Regression | Baseline Prediction |
| Random Forest Regressor | Ensemble Learning |
| Support Vector Regressor | Non-linear Prediction |

---

# 📈 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 📂 Project Structure

```text
AutoValueAI/
│
├── Data/
│   └── car data.csv
│
├── app.py
├── dashboard.py
├── requirements.txt
├── README.md
│
└── notebooks/
```

---

# 🔌 API Example

## Request

```json
POST /predict

{
    "Year": 2018,
    "Present_Price": 8.5,
    "Driven_kms": 42000,
    "Fuel_Type": "Petrol",
    "Selling_type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0
}
```

## Response

```json
{
    "predicted_selling_price_lakh": 6.92
}
```

---

# 🎯 Future Enhancements

- Docker Deployment
- AWS / Azure Deployment
- CI/CD Pipeline
- User Authentication
- Explainable AI (SHAP)
- Model Monitoring
- Database Integration
- Mobile Responsive UI

---

# 👨‍💻 Author

**Zubair Hamid**

**AI Engineer | Data Scientist | Machine Learning Enthusiast**

📧 Email: zubair.hamiid@gmail.com

🔗 LinkedIn: *(https://www.linkedin.com/in/zubair-hamid-073ba1332/)*

---

# 🌟 Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It helps support future open-source AI and Machine Learning projects.

---
