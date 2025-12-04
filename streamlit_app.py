import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Load Your Safe Model (Pickled)
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("airbnb_streamlit_safe.pkl")

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

neighbourhood = st.selectbox("Neighbourhood", [
    "Allied Gardens", "Alta Vista", "Amphitheater And Water Park", "Balboa Park", "Bario Logan",
    "Bay Ho", "Bay Park", "Bay Terrace", "Bird Land", "Bonita Long Canyon",
    "Carmel Mountain", "Carmel Valley", "Chollas View", "City Heights East",
    "City Heights West", "Clairemont Mesa East", "Clairemont Mesa West",
    "College Area", "College West", "Columbia", "Core-columbia", "Corridor",
    "Cortez", "Del Mar Heights", "Del Mar Mesa", "East Village",
    "Eastlake Trails", "Eastlake Vistas", "Eastlake Woods", "Egger Highlands",
    "El Cerritos", "Emerald Hills", "Encanto", "Estlake Greens",
    "Gaslamp Quarter", "Gateway", "Grant Hill", "Grantville", "Horton Plaza",
    "Jomacha-Lomita", "Kearny Mesa", "La Jolla", "La Jolla Village",
    "Lakeside", "Liberty Station", "Lincoln Park", "Little Italy",
    "Loma Portal", "Marina", "Memorial", "Middletown", "Midtown", "Mira Mesa",
    "Miramar", "Mission Bay Park", "Mission Beach", "Mission Valley East",
    "Mission Valley West", "Morena", "Mountain View", "Nestor", "Normal Heights",
    "North City", "North Clairemont", "North Park", "Ocean Beach", "Old Town",
    "Otay Mesa", "Otay Mesa West", "Pacific Beach", "Palm City", "Paradise Hills",
    "Paseo Del Sol", "Point Loma Heights", "Rancho Bernardo", "Rancho Encantada",
    "Rancho Penasquitos", "Rancho San Diego", "Ridley", "Rio Vista",
    "Rolando", "Rolando Park", "San Carlos", "San Pasqual", "San Ysidro",
    "Scripps Ranch", "Serra Mesa", "Shelltown", "Sorrento Valley",
    "South Park", "Southeastern San Diego", "Skyline", "Talmadge",
    "Tierrasanta", "University City", "University Heights", "Valencia Park",
    "Webster"
])



# ------------------------------
# Convert Inputs → Model Vector
# ------------------------------
def encode_inputs():
    # Base data row
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

    # One-hot encode the two categorical fields
    df = pd.get_dummies(df, columns=["room_type", "neighbourhood_cleansed"], drop_first=False)

    # Ensure columns align with training
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Arrange in correct order
    df = df[model.feature_names_in_]
    return df



# ------------------------------
# Predict Button
# ------------------------------
if st.button("Predict Nightly Price"):
    X = encode_inputs()
    prediction = model.predict(X)[0]
    st.success(f"Estimated Price: **${prediction:,.2f}**")
