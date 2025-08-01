
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory and the model file into the container
# Ensure your trained model is at src/model/flight_price_pipeline.pkl
COPY ./src/. .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Command to run the uvicorn server for the FastAPI app
# We use 0.0.0.0 to make it accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]