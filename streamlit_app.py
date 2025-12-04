import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    # IMPORTANT:
    # This must match the actual file name in your Streamlit repo.
    return joblib.load("airbnb_nn_model.pkl")

model = load_model()

# ------------------------------
# UI Title & Description
# ------------------------------
st.title("🏡 Airbnb Price Predictor – San Diego")
st.write("Enter listing details below to estimate the nightly price.")

# ------------------------------
# Inputs
# ------------------------------
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, step=0.5)
accommodates = st.number_input("Accommodates", min_value=1, max_value=16, step=1)
minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, step=1)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=5000, step=1)
review_scores_rating = st.number_input("Review Score (1–100)", min_value=1.0, max_value=100.0, step=1.0)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

room_type = st.selectbox("Room Type", [
    "Entire home/apt",
    "Private room",
    "Shared room",
    "Hotel room"
])

neighbourhood = st.text_input("Neighbourhood (exact name)", "")

# ------------------------------
# Convert Inputs → Model Vector
# ------------------------------
def encode_inputs():
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

    df = pd.get_dummies(df, columns=["room_type", "neighbourhood_cleansed"], drop_first=False)

    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    df = df[model.feature_names_in_]

    return df

# ------------------------------
# Predict Button
# ------------------------------
if st.button("Predict Nightly Price"):
    X = encode_inputs()
    pred = model.predict(X)[0]
    st.success(f"Estimated Price: **${pred:,.2f}**")
