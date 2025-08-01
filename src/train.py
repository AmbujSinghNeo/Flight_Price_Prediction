import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, max_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import argparse
from pathlib import Path

# --- Robust Path Configuration ---
# The path to the directory containing this script (src/)
SCRIPT_DIR = Path(__file__).resolve().parent
# The path to the data file, which is one level up from src/
RAW_DATA_PATH = SCRIPT_DIR.parent / "Clean_Dataset.csv"
# The path for the MLflow database, relative to the main project folder
MLFLOW_TRACKING_URI = f"sqlite:///{SCRIPT_DIR.parent / 'mlflow.db'}"

# --- Other Configuration ---
EXPERIMENT_NAME = "FlightPricePredictor"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "flight_price_pipeline.pkl")


# --- Model Definitions ---
def get_model(model_name, params):
    """Returns a model instance based on the name."""
    if model_name == "xgboost":
        return xgb.XGBRegressor(**params)
    elif model_name == "random_forest":
        return RandomForestRegressor(**params)
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# --- Main Training Function ---
def train_model(model_name, params):
    """
    Loads data, preprocesses it, trains a specified model,
    logs everything with MLflow, and saves the final pipeline.
    """
    print("--- Starting Model Training ---")
    print(f"Model: {model_name}, Params: {params}")

    # 1. Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load and Split Data
    print("Loading and splitting data...")
    # Assuming your preprocessing script is in the same src directory
    from preprocessing import create_preprocessing_pipeline
    
    df_raw = pd.read_csv(RAW_DATA_PATH)
    if 'Unnamed: 0' in df_raw.columns:
        df_raw = df_raw.drop('Unnamed: 0', axis=1)

    X = df_raw.drop('price', axis=1)
    y = np.log1p(df_raw['price']) # Apply log transformation to the target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Create the full pipeline (preprocessor + model)
    print("Creating the full model pipeline...")
    preprocessor = create_preprocessing_pipeline()
    regressor = get_model(model_name, params)
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    # 4. Train and Evaluate with MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)
        
        print("Training the pipeline...")
        model_pipeline.fit(X_train, y_train)
        
        y_pred = model_pipeline.predict(X_test)
        
        # Calculate all performance metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        
        print("\n--- Evaluation Metrics ---")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.4f}")
        print(f"  Max Error: {max_err:.4f}")
        print("--------------------------\n")

        # Log all metrics to MLflow
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("max_error", max_err)
        
        mlflow.sklearn.log_model(model_pipeline, "model_pipeline")
        
        print(f"MLflow run completed. Run ID: {run.info.run_id}")

    # Save the final pipeline for demonstration.
    print("Saving the trained pipeline...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Pipeline saved to: {MODEL_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flight price prediction model.")
    parser.add_argument("--model_name", type=str, default="xgboost", help="Model to train: 'xgboost', 'random_forest', or 'lightgbm'")
    
    # Common Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=7, help="Maximum depth of the trees.")
    
    # XGBoost/LightGBM specific
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for boosting models.")
    
    # LightGBM specific
    parser.add_argument("--num_leaves", type=int, default=31, help="Maximum number of leaves in one tree (for LightGBM).")

    args = parser.parse_args()

    # Collect relevant parameters for the chosen model
    if args.model_name == "xgboost":
        model_params = {
            'n_estimators': args.n_estimators,
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'random_state': 42,
            'objective': 'reg:squarederror'
        }
    elif args.model_name == "random_forest":
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'random_state': 42,
            'n_jobs': -1 # Use all available cores
        }
    elif args.model_name == "lightgbm":
        model_params = {
            'n_estimators': args.n_estimators,
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'num_leaves': args.num_leaves,
            'random_state': 42,
            'n_jobs': -1
        }
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
    
    train_model(args.model_name, model_params)

'''

### Step 2: Run Your Experiments

Now, you can run different experiments directly from your terminal without changing any code. Each time you run the script with different arguments, MLflow will log it as a new "run".

Navigate to your `src` folder in the terminal and run the following commands.

**Experiment 1: Default XGBoost**
```bash
python train.py
```

**Experiment 2: XGBoost with more estimators**
```bash
python train.py --n_estimators 200
```

**Experiment 3: XGBoost with a deeper tree**
```bash
python train.py --max_depth 10 --n_estimators 150
```

**Experiment 4: Switch to Random Forest**
```bash
python train.py --model_name random_forest --n_estimators 100 --max_depth 10
```

**Experiment 5: Another Random Forest**
```bash
python train.py --model_name random_forest --n_estimators 200 --max_depth 12
```

### Step 3: View and Compare Results in the MLflow UI

This is the most powerful part. After you've run a few experiments, you can launch the MLflow User Interface to see and compare everything.

1.  **Launch the UI:** In the same terminal (from your main project folder, not `src`), run the following command:
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```
    This tells MLflow to launch a web server and use your `mlflow.db` file as the source of truth.

2.  **Open Your Browser:** Open a web browser and go to `http://127.0.0.1:5000` (or `http://localhost:5000`).

3.  **Analyze Your Results:** You will see a dashboard with a table listing all your runs.
    * **Columns:** You'll see columns for the parameters you logged (`model_name`, `max_depth`, `n_estimators`, etc.) and the metrics you logged (`r2_score`, `rmse`).
    * **Find the Best Model:** Simply **click on the `r2_score` column header** to sort all your runs from best to worst. This immediately tells you which combination of model and hyperparameters performed the best!
    * **Compare Runs:** You can select multiple runs by clicking the checkboxes next to them and then click the **"Compare"** button. MLflow will show you a detailed side-by-side comparison of their parameters and metrics.

This workflow allows you to systematically experiment, track everything automatically, and easily find the best model to depl


### How to Proceed: Your Step-by-Step Guide

1.  **Create the Directory Structure:**
    * Create a main project folder (e.g., `FlightPricePredictor`).
    * Inside it, create a `src` folder.
    * Inside `src`, create a `model` folder.
    * Place your `flight_price.csv` in a `data` folder at the same level as `src`.
    * Place your EDA notebook in a `notebooks` folder.
    * Your `preprocessing.py` and the new `train.py` will go inside `src`.

2.  **Run the Training Script:**
    * Install the necessary libraries (`mlflow`, `xgboost`, `scikit-learn`, etc.).
    * From your terminal, navigate into the `src` folder and run the command:
        ```bash
        python train.py
        ```

3.  **Check the Outputs:**
    * A new file, `mlflow.db`, will be created to store your experiment results.
    * A `mlruns` folder will appear, containing the detailed logs for each run.
    * Most importantly, your trained pipeline will be saved as **`src/model/flight_price_pipeline.pkl`**.

4.  **Next Steps (After Training):**
    * **`src/schema.py`**: Define your Pydantic model for the API input.
    * **`src/utils.py`**: Create a function to load the `flight_price_pipeline.pkl` file.
    * **`src/main.py`**: Build the FastAPI app. It will import the loader from `utils.py` and the schema from `schema.py` to create the `/predict` endpoint.

You are now perfectly set up to train your first model and transition into the API development pha

'''