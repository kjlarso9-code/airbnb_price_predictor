import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    model = joblib.load("airbnb_nn_model.pkl")   # Make sure file is in the repo
    return model

model = load_model()

st.title("🏡 Airbnb Price Predictor")
st.write("Enter listing details below to estimate the nightly price.")

# ==========================
# USER INPUTS
# ==========================
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, step=0.5)
accommodates = st.number_input("Accommodates", min_value=0, max_value=20, step=1)
minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, step=1)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=5000, step=1)
review_scores_rating = st.number_input("Review Score Rating", min_value=20.0, max_value=100.0, step=1.0)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

room_type = st.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
)

neighbourhood = st.text_input("Neighbourhood", "Unknown")

# ==========================
# ENCODING INPUTS
# ==========================
def format_input_for_model():
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

    df = pd.DataFrame([data])

    # One-hot encode same as training
    df = pd.get_dummies(df, columns=["room_type", "neighbourhood_cleansed"], drop_first=False)

    # Add missing columns
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Order columns correctly
    df = df[model.feature_names_in_]

    return df

# ==========================
# PREDICTION BUTTON
# ==========================
if st.button("Predict Price"):
    X = format_input_for_model()
    prediction = model.predict(X)[0]
    st.success(f"Estimated Nightly Price: **${prediction:,.2f}**")
