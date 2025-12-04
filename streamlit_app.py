import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("🏡 Airbnb Price Predictor")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("airbnb_nn_model.pkl")

model = joblib.load("airbnb_streamlit_safe.pkl")

# User inputs
bedrooms = st.number_input("Bedrooms", 0, 10, 1)
bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0)
accommodates = st.number_input("Accommodates", 1, 16, 2)
minimum_nights = st.number_input("Minimum Nights", 1, 365, 1)
number_of_reviews = st.number_input("Number of Reviews", 0, 5000, 10)
review_scores_rating = st.number_input("Review Score Rating", 0.0, 100.0, 90.0)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

room_type = st.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
)

neighbourhood = st.text_input("Neighbourhood Cleansed")

# Prepare input
def make_input():
    data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "accommodates": accommodates,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "review_scores_rating": review_scores_rating,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "neighbourhood_cleansed": neighbourhood
    }
    return pd.DataFrame([data])

if st.button("Predict Price"):
    X = make_input()
    y_pred = model.predict(X)[0]
    st.success(f"Estimated nightly price: **${y_pred:,.2f}**")
