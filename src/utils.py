import joblib
from pathlib import Path
import os
import logging

# Get the same logger that main.py uses
logger = logging.getLogger(__name__)

# This script provides helper functions, such as loading model artifacts.

# Define the path to the model artifact relative to this script's location
BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_PATH = BASE_DIR / "model/flight_price_pipeline.pkl"

def load_model():
    """
    Loads the trained model pipeline from the specified path.
    """
    # --- THIS IS THE NEW LOGGING DEBUG ---
    logger.info(f"---!!! ATTEMPTING TO LOAD MODEL FROM: {MODEL_PATH} !!!---")
    logger.info(f"---!!! DOES THE FILE EXIST? {'YES' if os.path.exists(MODEL_PATH) else 'NO'} !!!---")
    # ------------------------------------

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from utils.")
        return model
    except Exception as e:
        logger.error(f"---!!! AN ERROR OCCURRED IN load_model() !!!---")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e}")
        return None

# Example of running the loader directly (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # Configure logging for direct run
    loaded_model = load_model()
    if loaded_model:
        print(f"Model object: {loaded_model}")
