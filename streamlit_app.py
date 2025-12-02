import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("airbnb_model_small.pkl")

# The 12 features your model was trained on:
feature_columns = [
    "bedrooms",
    "bathrooms",
    "accommodates",
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "room_type_Hotel room",
    "room_type_Private room",
    "room_type_Shared room",
    "neighbourhood_Tijuana, Baja California, Mexico"
]

st.title("🏡 Airbnb Price Predictor")
st.write(
    """
    Enter listing details to estimate nightly price.  
    **Model trained on San Diego Airbnb dataset.**
    """
)

st.subheader("Listing Features")

inputs = {}

# Numeric fields
inputs["bedrooms"] = st.number_input("Bedrooms", min_value=0.0, value=1.0)
inputs["bathrooms"] = st.number_input("Bathrooms", min_value=0.0, value=1.0)
inputs["accommodates"] = st.number_input("Accommodates", min_value=1, value=2)

inputs["latitude"] = st.number_input("Latitude", value=32.7157)
inputs["longitude"] = st.number_input("Longitude", value=-117.1611)

inputs["minimum_nights"] = st.number_input("Minimum Nights", min_value=1, value=1)
inputs["number_of_reviews"] = st.number_input("Number of Reviews", min_value=0, value=10)
inputs["review_scores_rating"] = st.number_input(
    "Review Score (0–5)", min_value=0.0, max_value=5.0, value=4.5
)

# One-hot encoded room_type
room_type = st.selectbox(
    "Room Type",
    ["Entire home/apt", "Hotel room", "Private room", "Shared room"]
)

inputs["room_type_Hotel room"] = 1 if room_type == "Hotel room" else 0
inputs["room_type_Private room"] = 1 if room_type == "Private room" else 0
inputs["room_type_Shared room"] = 1 if room_type == "Shared room" else 0

# One-hot encoded neighbourhood
neighbourhood = st.selectbox(
    "Neighbourhood",
    ["San Diego", "Tijuana, Baja California, Mexico"]
)

inputs["neighbourhood_Tijuana, Baja California, Mexico"] = (
    1 if neighbourhood == "Tijuana, Baja California, Mexico" else 0
)

# Predict
if st.button("Predict Price"):
    X = np.array([[inputs[col] for col in feature_columns]])
    prediction = model.predict(X)[0]

    st.success(f"Estimated Nightly Price: **${prediction:.2f}**")

    st.caption(
        """
        *Note: Estimate based on historical data.  
        Use as a guide, not a guaranteed price.*
        """
    )

