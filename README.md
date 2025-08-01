# Flight Price Predictor API & Dashboard

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![Python Version](https://img.shields.io/badge/python-3.9-blue) ![Docker](https://img.shields.io/badge/docker-ready-blue)

A complete MLOps project to predict flight prices. This repository contains a machine learning model served via a **FastAPI** backend, an interactive **Streamlit** dashboard, and is fully containerized with **Docker**.

## Features

- **REST API:** A robust API built with FastAPI to serve real-time price predictions.
- **Interactive Dashboard:** A user-friendly frontend built with Streamlit to interact with the model.
- **MLOps Ready:** Integrated with **MLflow** for experiment tracking and model versioning.
- **Containerized:** Fully containerized with **Docker** and orchestrated with **Docker Compose** for easy setup and deployment.
- **CI/CD Ready:** Configured for automated builds and deployments using GitHub Actions.

## Tech Stack

- **Backend:** Python, FastAPI, Scikit-learn, XGBoost
- **Frontend:** Streamlit
- **MLOps & Deployment:** Docker, Docker Compose, MLflow, GitHub Actions

## Project Architecture

This project follows a microservice-based architecture. The diagram below illustrates the complete workflow from user interaction to price prediction.

```mermaid
graph TD
    subgraph User's Browser
        A[User Enters Flight Details] --> B{Streamlit UI};
        B --> C[Clicks 'Predict Price'];
    end

    subgraph "Docker Network"
        C -->|1. HTTP POST Request| D[FastAPI Backend];
        D -->|2. Load Model| E[ML Model];
        D -->|3. Preprocess Input| F[Preprocessing Pipeline];
        F --> G[Make Prediction];
        G -->|4. Return Prediction| D;
        D -->|5. HTTP JSON Response| C;
    end

    subgraph User's Browser
        C --> H{Display Predicted Price};
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px

Getting Started
Follow these steps to get the project running locally.

Prerequisites
Docker

Docker Compose

Installation & Setup
Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Train the Model: Before running the application, you must train a model. The trained model artifact (.pkl) is required by the API.

# Navigate to the source directory
cd src

# Run the training script (this will create the model in src/model/)
python train.py

# Go back to the root directory
cd ..

Run the Application: Use Docker Compose to build and run the containers.

docker-compose up --build -d

Access the Services:

Streamlit Dashboard: Open your browser to http://localhost:8501

API Docs: Access the interactive API documentation at http://localhost:8000/docs

API Usage
You can send a POST request to the /predict endpoint to get a price prediction.

Endpoint: http://localhost:8000/predict

Request Body:

{
  "airline": "Vistara",
  "source_city": "Delhi",
  "departure_time": "Morning",
  "stops": "one",
  "arrival_time": "Night",
  "destination_city": "Mumbai",
  "class": "Business",
  "duration": 15.83,
  "days_left": 26
}

Success Response:

{
  "predicted_price": 85000.50
}
