🏡 Airbnb Price Predictor

Machine Learning • Streamlit App • Databricks • End-to-End Deployment

📌 Project Overview

This project predicts nightly Airbnb prices using machine learning models trained on a San Diego Airbnb dataset. It includes:

Full data cleaning + preprocessing

Feature engineering

Model development (baseline → Random Forest)

MLflow tracking in Databricks

A deployed Streamlit web app that allows users to enter listing details and receive a predicted price

End-to-end workflow demonstration

GitHub repository + Render deployment (app link included)

🎯 Business Problem

Airbnb hosts often struggle to set the correct price—too low means lost revenue, too high means fewer bookings.
This project helps solve:

What should a host charge per night?

How do listing features influence price?

Which neighborhoods, room types, or amenities increase value?

Accurate price predictions help hosts optimize revenue and maximize occupancy.

🧹 Data Cleaning & Preparation

Steps performed in Databricks:

Removed missing or invalid entries (bathrooms, review scores, bedrooms, neighborhood)

Converted price column to numeric

One-hot encoded categorical fields:

room_type

neighbourhood

Standardized numerical values

Selected final model features:

bedrooms
bathrooms
accommodates
latitude
longitude
minimum_nights
number_of_reviews
review_scores_rating
room_type_Hotel room
room_type_Private room
room_type_Shared room
neighbourhood_Tijuana, Baja California, Mexico

🤖 Model Development
Baseline Model

Linear Regression

MAE: (insert from your results)

R² Score: (insert)

Advanced Model — Random Forest Regressor

200 estimators

Random state = 42

MAE: (insert)

R² Score: (insert)

Tracked using MLflow

Best performance among tested models

Why Random Forest Won

Handles nonlinear relationships

Less sensitive to outliers

Performs well on tabular Airbnb-style datasets

Requires minimal feature scaling

🖥 Deployment (Streamlit App)

The Streamlit app accepts user inputs such as:

Bedrooms

Bathrooms

Accommodates

Latitude & longitude

Minimum nights

Number of reviews

Review score

Room type

Neighborhood

Then predicts:

🎯 Estimated nightly price

The UI includes notes on limitations, model confidence, and how predictions should be interpreted.

🌐 Links

🔗 Deployed App (Render): (add your link here)
📘 Databricks Workspace / MLflow: (https://dbc-65647401-6b36.cloud.databricks.com/editor/notebooks/2028523477733230?o=1081562564116675)
🗂 GitHub Repository: (this repo)

📁 Repository Structure
airbnb_price_predictor/
│
├── app.py                # Streamlit application
├── requirements.txt      # Dependencies for Render
├── airbnb_model_small.pkl    # Saved Random Forest model
├── README.md             # Project documentation
└── /notebooks            # Databricks notebooks (optional)

🧠 How the App Works

The model expects a 12-feature input vector.
Streamlit reconstructs this vector from user selections by:

Collecting numerical inputs

Converting categorical selections to one-hot encoded format

Feeding the array into the trained Random Forest model

Output is displayed in a clean, user-friendly UI.

🎥 Presentation Requirements (Project Final Deliverables)

This project includes:

✔ Business framing
✔ Data cleaning & EDA
✔ Model development + comparison
✔ Best model justification
✔ Deployment demo (Streamlit app)
✔ End-to-end workflow
✔ Repository documentation

🙋‍♀️ Author

Kendall Larson
CIS 508 – Final Project
Instructor: Sang-Pil Han
