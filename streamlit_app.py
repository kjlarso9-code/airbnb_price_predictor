import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("airbnb_model_small.pkl")

model = load_model()

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🏡 Airbnb Price Predictor")
st.write("Enter your Airbnb listing details to estimate the nightly price.")

# -------------------------------
# User Inputs
# -------------------------------

bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
accommodates = st.number_input("Accommodates", min_value=1, max_value=16, value=2)
minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=30, value=1)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=10)
review_scores_rating = st.number_input("Review Score Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)

latitude = st.number_input("Latitude", value=32.7157)
longitude = st.number_input("Longitude", value=-117.1611)

room_type = st.selectbox("Room Type", [
    "Entire home/apt",
    "Private room",
    "Shared room",
    "Hotel room"
])

neighbourhood_cleansed = st.text_input("Neighborhood", "Downtown")

# -------------------------------
# Predict button
# -------------------------------

if st.button("Predict Price"):

    # Create a dataframe to match model features
    input_data = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "accommodates": accommodates,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "review_scores_rating": review_scores_rating,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "neighbourhood_cleansed": neighbourhood_cleansed
    }])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    st.success(f"### Estimated Price: **${prediction:,.2f}** per night")

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.write("Built by Kendall Larson — CIS 508 Final Project")
