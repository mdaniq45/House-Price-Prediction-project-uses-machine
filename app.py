import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("house_price_model.pkl")

# App title
st.title("House Price Prediction App")

# User inputs
st.header("Enter the house details:")
area = st.number_input("Area (Location Code):", min_value=0, max_value=10, step=1)
size = st.number_input("Size (in sqft):", min_value=100, max_value=10000, step=100)
bedrooms = st.number_input("Number of Bedrooms:", min_value=1, max_value=10, step=1)
price_per_sqft = st.number_input("Price per sqft:", min_value=1, max_value=1000, step=1)

# Button for prediction
if st.button("Predict Price"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        "Area": [area],
        "Size": [size],
        "Bedrooms": [bedrooms],
        "Price_per_sqft": [price_per_sqft]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success(f"The predicted house price is ₹{prediction:,.2f}")

# Footer
st.write("Developed with ❤️ using Streamlit.")
