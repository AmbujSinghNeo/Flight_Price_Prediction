from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import pandas as pd
import numpy as np
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from sklearn.pipeline import Pipeline

# Import the schema and utility functions from other modules
from schema import FlightPredictionInput, FlightPredictionOutput
from utils import load_model

# --- Application Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading and Dependency ---
# Load the model once at startup and provide it through a dependency function.
pipeline = load_model()

def get_model_pipeline():
    """Dependency function to get the loaded model pipeline."""
    return pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for logging startup/shutdown."""
    logger.info("Application startup...")
    if pipeline:
        logger.info("Model pipeline loaded successfully at startup.")
    else:
        logger.error("FATAL: Model pipeline failed to load at startup.")
    yield
    logger.info("Application shutdown.")

# --- Create the FastAPI App Instance ---
app = FastAPI(
    title="Flight Price Predictor API",
    description="An API to predict flight prices using a trained ML model.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Custom Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_messages = []
    for error in exc.errors():
        field = " -> ".join(map(str, error["loc"]))
        message = error["msg"]
        error_messages.append(f"Error in field '{field}': {message}")
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"error": "Validation Error", "detail": ". ".join(error_messages)})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"An unexpected error occurred: {exc}", exc_info=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Internal Server Error", "detail": "An unexpected internal error occurred."})

# --- API Endpoints ---
@app.get("/", tags=["Status"])
def read_root() -> Dict[str, str]:
    return {"status": "ok", "message": "Welcome to the Flight Price Predictor API!"}

@app.post("/predict", response_model=FlightPredictionOutput, tags=["Prediction"])
def predict_price(input_data: FlightPredictionInput, model: Pipeline = Depends(get_model_pipeline)):
    """Predicts the flight price based on input features."""
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The model is currently unavailable.")
    
    try:
        input_df = pd.DataFrame([input_data.model_dump(by_alias=True)])
        prediction_log = model.predict(input_df)
        predicted_price = np.expm1(prediction_log[0])
        
        if not np.isfinite(predicted_price) or predicted_price <= 0:
            raise ValueError("Model produced an invalid (non-positive or non-finite) price.")
        
        logger.info(f"Prediction successful. Price: ₹{predicted_price:.2f}")
        return FlightPredictionOutput(predicted_price=predicted_price)

    except ValueError as ve:
        logger.warning(f"Prediction failed due to invalid data: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"An unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred while processing the prediction.")
