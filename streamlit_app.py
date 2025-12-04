import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    return joblib.load("nn_reg_only.pkl")

model = load_model()

st.title("🏡 Airbnb Price Predictor")

# INPUTS
bedrooms = st.number_input("Bedrooms", 0, 10)
bathrooms = st.number_input("Bathrooms", 0.0, 10.0, step=0.5)
accommodates = st.number_input("Accommodates", 0, 16)
minimum_nights = st.number_input("Minimum Nights", 1, 365)
number_of_reviews = st.number_input("Number of Reviews", 0, 2000)
review_scores_rating = st.number_input("Review Score Rating", 20.0, 100.0)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

room_type = st.selectbox("Room Type", ["Hotel room","Private room","Shared room","Entire home/apt"])
neighbourhood = st.text_input("Neighborhood Cleansed")

# FEATURE ENCODING
def encode():
    df = pd.DataFrame([{
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
    }])

    df = pd.get_dummies(df, columns=["room_type", "neighbourhood_cleansed"], drop_first=False)

    expected = ['bedrooms','bathrooms','accommodates','minimum_nights',
                'number_of_reviews','review_scores_rating','latitude','longitude']

    # add dummy columns
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    df = df[model.feature_names_in_]

    return df

# PREDICT
if st.button("Predict Price"):
    X = encode()
    pred = model.predict(X)[0]
    st.success(f"Estimated Nightly Price: **${pred:,.2f}**")
