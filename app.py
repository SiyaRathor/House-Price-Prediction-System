import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and columns
model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏠 House Price Prediction App")
st.write("Enter house details:")

# ---- USER INPUTS ----
overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sq ft)", value=1500)
garage_cars = st.number_input("Garage Cars", value=1)
year_built = st.number_input("Year Built", value=2000)

# ---- CREATE EMPTY INPUT WITH SAME STRUCTURE ----
input_df = pd.DataFrame(columns=model_columns)
input_df.loc[0] = 0

# ---- FILL USER VALUES ----
if "OverallQual" in input_df.columns:
    input_df.at[0, "OverallQual"] = overall_qual

if "GrLivArea" in input_df.columns:
    input_df.at[0, "GrLivArea"] = gr_liv_area

if "GarageCars" in input_df.columns:
    input_df.at[0, "GarageCars"] = garage_cars

if "YearBuilt" in input_df.columns:
    input_df.at[0, "YearBuilt"] = year_built

# ---- PREDICT ----
if st.button("Predict Price"):
    prediction_log = model.predict(input_df)[0]
    price = np.expm1(prediction_log)
    st.success(f"Estimated House Price: ${int(price):,}")
