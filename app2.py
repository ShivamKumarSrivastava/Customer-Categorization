import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Cluster Prediction",
    layout="centered"
)

st.title("🧠 Customer Cluster Prediction")
st.write("Fill in customer details to predict the cluster")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("artifacts/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --------------------------------------------------
# Encoding maps (MUST match training)
# --------------------------------------------------
education_map = {
    "Basic": 0,
    "Graduation": 1,
    "Master": 2,
    "PhD": 3
}

marital_map = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

parental_map = {
    "No": 0,
    "Yes": 1
}

# --------------------------------------------------
# Input Form
# --------------------------------------------------
with st.form("customer_form"):
    Age = st.number_input("Age", min_value=18, max_value=100)

    Education = st.selectbox(
        "Education", list(education_map.keys())
    )

    Marital_Status = st.selectbox(
        "Marital Status", list(marital_map.keys())
    )

    Parental_Status = st.selectbox(
        "Parental Status", list(parental_map.keys())
    )

    Children = st.number_input("Children", min_value=0, max_value=10)
    Income = st.number_input("Income", min_value=0)
    Total_Spending = st.number_input("Total Spending", min_value=0)
    Days_as_Customer = st.number_input("Days as Customer", min_value=0)
    Recency = st.number_input("Recency", min_value=0)

    Wines = st.number_input("Wines Spending", min_value=0)
    Fruits = st.number_input("Fruits Spending", min_value=0)
    Meat = st.number_input("Meat Spending", min_value=0)
    Fish = st.number_input("Fish Spending", min_value=0)
    Sweets = st.number_input("Sweets Spending", min_value=0)
    Gold = st.number_input("Gold Spending", min_value=0)

    Web = st.number_input("Web Purchases", min_value=0)
    Catalog = st.number_input("Catalog Purchases", min_value=0)
    Store = st.number_input("Store Purchases", min_value=0)

    Discount_Purchases = st.number_input("Discount Purchases", min_value=0)
    Total_Promo = st.number_input("Total Promo Purchases", min_value=0)
    NumWebVisitsMonth = st.number_input("Web Visits per Month", min_value=0)

    submit = st.form_submit_button("Predict Cluster")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submit:
    try:
        # Encode categorical features
        Education_val = education_map[Education]
        Marital_Status_val = marital_map[Marital_Status]
        Parental_Status_val = parental_map[Parental_Status]

        # Create numeric input (ORDER MUST MATCH TRAINING)
        input_data = np.array([[
            Age,
            Education_val,
            Marital_Status_val,
            Parental_Status_val,
            Children,
            Income,
            Total_Spending,
            Days_as_Customer,
            Recency,
            Wines,
            Fruits,
            Meat,
            Fish,
            Sweets,
            Gold,
            Web,
            Catalog,
            Store,
            Discount_Purchases,
            Total_Promo,
            NumWebVisitsMonth
        ]])

        prediction = model.predict(input_data)

        st.success(
            f"🎯 Predicted Customer Cluster: {int(prediction[0])}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
