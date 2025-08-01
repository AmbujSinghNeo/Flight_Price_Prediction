import mlflow
from pathlib import Path
import logging
import os

# Get the same logger that main.py uses
logger = logging.getLogger(__name__)

# --- MLflow Configuration ---
# Define the path to the MLflow tracking database relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
# The tracking URI should point to the mlflow.db file in the /app directory
MLFLOW_TRACKING_URI = f"sqlite:///{SCRIPT_DIR / 'mlflow.db'}"

# Define the name of the model in the MLflow Model Registry
REGISTERED_MODEL_NAME = "flight-price-predictor"

def load_model():
    """
    Loads the production model pipeline from the MLflow Model Registry.
    """
    try:
        # Set the MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Define the model URI to load the version with the 'production' alias
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"
        
        logger.info(f"---!!! ATTEMPTING TO LOAD PRODUCTION MODEL FROM MLFLOW: {model_uri} !!!---")
        
        # Load the model from the registry
        model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info("Production model loaded successfully from MLflow Model Registry.")
        return model
        
    except Exception as e:
        logger.error(f"---!!! AN ERROR OCCURRED WHILE LOADING MODEL FROM MLFLOW !!!---")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e}")
        logger.error(
            "Please ensure the MLflow tracking server is accessible and a model version "
            f"named '{REGISTERED_MODEL_NAME}' has been assigned the 'production' alias."
        )
        return None

# Example of running the loader directly (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # Configure logging for direct run
    
    # Before running this test, make sure you have assigned the 'production'
    # alias to one of your model versions in the MLflow UI.
    loaded_model = load_model()
    
    if loaded_model:
        print("\n--- Model Loaded Successfully ---")
        print(f"Model Object Type: {type(loaded_model)}")
