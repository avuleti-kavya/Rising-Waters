# 🌊 Rising Waters – ML-Powered Flood Prediction System

A machine learning web application that predicts flood risk based on meteorological data, built with Flask and deployed via a clean browser-based UI.

## Overview

Floods are among the most devastating natural disasters. This system addresses the lack of timely early-warning tools by training classification models on historical weather data and serving predictions through a Flask web application.

## ML Models Used

| Model | Accuracy |
|---|---|
| Decision Tree | 99.50% |
| Random Forest | 99.50% |
| KNN | 93.00% |
| **XGBoost** | **100.00%** ✅ |

The best-performing model (XGBoost) is automatically saved and used for predictions.

## Features

- Predicts flood risk from 6 meteorological inputs
- Displays flood probability percentage
- Visual charts: accuracy comparison, feature importance, confusion matrix
- REST API endpoint (`/api/predict`) for programmatic access
- Designed for IBM Cloud deployment

## Input Features

- Annual Rainfall (mm)
- Cloud Visibility (km)
- Monsoon Rainfall (mm)
- Pre-Monsoon Rainfall (mm)
- Post-Monsoon Rainfall (mm)
- Winter Rainfall (mm)

## Project Structure

```
Rising Waters/
├── app.py                  # Flask web application
├── flood_model.py          # Model training + chart generation
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web UI
└── static/                 # Generated chart images
    ├── accuracy_comparison.png
    ├── feature_importance.png
    └── confusion_matrix.png
```

## Getting Started

```bash
pip install -r requirements.txt
python flood_model.py   # Train models & generate charts
python app.py           # Start Flask at http://127.0.0.1:5000
```

## Scenarios

- **Early Warning**: Meteorologists enter live readings to trigger evacuation advisories
- **Resource Allocation**: Coordinators monitor multiple regions during monsoon season
- **Model Validation**: Analysts test against historical data to confirm reliability

## Tech Stack

Python · Flask · Scikit-Learn · XGBoost · NumPy · Pandas · Matplotlib · Seaborn
