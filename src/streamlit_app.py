import streamlit as st
import requests
import os

# Import configurations from the central config file
# Make sure you have a 'config.py' file in the same directory
from config import API_ENDPOINT, AIRLINES, CITIES, TIMES, STOPS, CLASSES

# --- Page Configuration ---
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("✈️ Flight Price Predictor")
st.markdown(
    "Welcome! This application uses a machine learning model to predict flight prices. "
    "Use the controls in the sidebar to enter your flight details."
)

# --- Sidebar for User Inputs ---
st.sidebar.header("Enter Flight Details")

# Create input widgets in the sidebar using imported lists
with st.sidebar:
    airline = st.selectbox("Airline", options=AIRLINES)
    source_city = st.selectbox("Source City", options=CITIES)
    destination_city = st.selectbox("Destination City", options=CITIES, index=1)
    
    st.markdown("---")
    
    departure_time = st.selectbox("Departure Time", options=TIMES)
    arrival_time = st.selectbox("Arrival Time", options=TIMES, index=1)
    
    st.markdown("---")
    
    stops = st.selectbox("Number of Stops", options=STOPS)
    class_type = st.selectbox("Class", options=CLASSES)
    
    st.markdown("---")
    
    duration = st.slider("Duration (hours)", min_value=0.5, max_value=48.0, value=15.0, step=0.5)
    days_left = st.slider("Days Left for Departure", min_value=1, max_value=50, value=26)
    
    st.markdown("---")

    # --- Client-side validation ---
    is_invalid_selection = source_city == destination_city
    if is_invalid_selection:
        st.warning("Source and Destination cities cannot be the same.")

    # Disable button if validation fails
    predict_button = st.button(
        "Predict Price", 
        type="primary", 
        use_container_width=True, 
        disabled=is_invalid_selection
    )

# --- Main Page Logic ---
if predict_button:
    # Create the payload dictionary from user inputs
    # Using "class_type" as the key for clarity and to avoid Python reserved keywords
    payload = {
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class_type": class_type,  # CHANGED KEY from "class"
        "duration": duration,
        "days_left": days_left
    }

    # Display a spinner while waiting for the API response
    with st.spinner('🚀 Sending data to the model for prediction...'):
        try:
            # Send a POST request to the FastAPI endpoint
            response = requests.post(API_ENDPOINT, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Display the prediction
            prediction_data = response.json()
            predicted_price = prediction_data.get("predicted_price")
            
            st.success("✅ Prediction Successful!")
            st.metric(label="Predicted Flight Price", value=f"₹ {predicted_price:,.2f}")
            
            with st.expander("See the data sent to the API"):
                st.json(payload)

        except requests.exceptions.HTTPError as http_err:
            # Handle API errors (like validation errors from FastAPI)
            st.error(f"API Error: Failed to get a prediction.")
            try:
                error_details = http_err.response.json()
                st.warning(f"Reason: {error_details.get('detail', 'No specific reason provided.')}")
            except requests.JSONDecodeError:
                st.warning("Could not parse the error response from the API.")

        except requests.exceptions.RequestException:
            # Handle connection errors
            st.error("Connection Error: Could not connect to the prediction service.")
            st.warning(
                "Please make sure the API server is running and accessible. "
                f"The app is trying to connect to: `{API_ENDPOINT}`"
            )