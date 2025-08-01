# Use the same base image for consistency
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files for the Streamlit app
# This includes the app itself and the config file it depends on
COPY ./src/streamlit_app.py .
COPY ./src/config.py .

# Make the Streamlit port available
EXPOSE 8501

# The command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]