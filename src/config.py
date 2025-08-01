# config.py

# This file acts as a single source of truth for model configurations.

# API Endpoint URL
# Use an environment variable in production, but a default for local development.
import os
API_ENDPOINT = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


# Feature Options (These should match the categories your model was trained on)
AIRLINES = ['Vistara', 'Air_India', 'Indigo', 'GO_FIRST', 'AirAsia', 'SpiceJet']
CITIES = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
TIMES = ['Morning', 'Early_Morning', 'Evening', 'Night', 'Afternoon', 'Late_Night']
STOPS = ['one', 'zero', 'two_or_more']
CLASSES = ['Economy', 'Business']

