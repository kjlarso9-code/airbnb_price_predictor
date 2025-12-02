# 🏡 Airbnb Price Predictor

Machine Learning • Streamlit App • Databricks • End-to-End Deployment

## 📌 Project Overview
This project predicts nightly Airbnb prices for San Diego listings using machine learning.

The project includes:
- Data cleaning & preprocessing (Databricks)
- Feature engineering
- Model selection & hyperparameter tuning
- MLflow experiment tracking
- Exporting the best model as a .pkl
- Deploying a Streamlit web app connected to GitHub

The final deliverable is a fully deployed interactive application that allows users to estimate Airbnb prices based on property characteristics.

## 🎯 Business Problem
Airbnb hosts must determine the optimal nightly price.

If the price is:
- Too low: revenue is lost
- Too high: bookings decrease

This project answers:
- What should a host charge per night?
- Which features most influence price?
- How do reviews, room type, and neighborhood affect value?

Accurate pricing helps hosts maximize occupancy and profit.

## 🧹 Data Cleaning & Preparation
Performed in Databricks:
- Cleaning
- Removed missing/invalid rows (bathrooms, bedrooms, scores)
- Converted price from string → numeric ($174.00 → 174.00)
- Filtered extreme outliers
- Feature Engineering

One-hot encoded:
- room_type
- neighbourhood

Standardized all numeric columns

Final model training features:
- bedrooms
- bathrooms
- accommodates
- latitude
- longitude
- minimum_nights
- number_of_reviews
- review_scores_rating
- room_type_Hotel room
- room_type_Private room
- room_type_Shared room
- neighbourhood_Tijuana, Baja California, Mexico

## 🤖 Model Development

Multiple models were tested and logged in MLflow:
### Baseline
- Linear Regression
- MAE: 447.11
- R²: 0.805
- RMSE: 1745.18
  
### Advanced Models
- Random Forest Regressor (final chosen model)
  - 200 estimators
  - random_state = 42
  - Best performance across RMSE/MAE
  - Robust to outliers and nonlinear patterns
    
### Why Random Forest Won
- Handles complex interactions
- Low overfitting risk
- Strong performance on tabular datasets
- No need for heavy scaling

Final model exported as:
✔ airbnb_model_small.pkl

## 🖥 Deployment (Streamlit App)
The Streamlit application:
- Accepts user inputs
- Builds a 12-feature vector
- Applies one-hot encoding logic
- Loads the trained model .pkl
- Returns an estimated nightly price

## 🔗 Live App
https://airbnbpricepredictor-9hepznyari5xotvfaz6z7e.streamlit.app/

📁 Repository Structure <br>
airbnb_price_predictor/ <br>
app.py = Streamlit application <br>
requirements.txt = Dependencies for Render <br>
airbnb_model_small.pkl = Saved Random Forest model <br>
README.md = Project documentation <br>
notebooks = Databricks notebooks <br>

## 🧠 How the App Works
1. User enters listing details
2. Streamlit preprocesses input (one-hot encoding, numeric conversion)
3. Data is passed into the Random Forest model
4. Prediction is displayed in an interactive UI

## 🚀 How to Run Locally
pip install -r requirements.txt <br>
streamlit run app.py

## 🙋‍♀️ Author

Kendall Larson <br>
CIS 508 – Term Project <br>
Instructor: Sang-Pil Han
