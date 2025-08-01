import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# --- Custom Transformers for Feature Engineering ---
# These remain the same as they are already modular.

class RouteTransformer(BaseEstimator, TransformerMixin):
    """
    Combines 'source_city' and 'destination_city' into a single 'route' feature
    and drops the original columns.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        # Drop flight column here as it's part of initial cleaning
        if 'flight' in X_.columns:
            X_ = X_.drop('flight', axis=1)
        X_['route'] = X_['source_city'] + '_' + X_['destination_city']
        X_ = X_.drop(['source_city', 'destination_city'], axis=1)
        return X_

class BookingUrgencyTransformer(BaseEstimator, TransformerMixin):
    """
    Creates 'booking_urgency' bins from 'days_left' and drops the original column.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        bins = [0, 3, 7, 15, 50]
        labels = ['Last_Minute', 'One_Week_Out', 'Two_Weeks_Out', 'Advance_Booking']
        X_['booking_urgency'] = pd.cut(X_['days_left'], bins=bins, labels=labels, right=True)
        X_ = X_.drop('days_left', axis=1)
        return X_

# --- Main Preprocessing Pipeline Creation Function ---

def create_preprocessing_pipeline():
    """
    Creates and returns a scikit-learn Pipeline that chains all feature
    engineering and encoding steps together.
    """
    # Define which columns will be one-hot encoded AFTER the custom transformations
    final_categorical_features = [
        'airline', 'departure_time', 'stops', 'arrival_time', 'class',
        'route', 'booking_urgency'
    ]

    # Create the ColumnTransformer to handle one-hot encoding
    # The remainder='passthrough' will keep the 'duration' column untouched.
    encoder = ColumnTransformer(
        transformers=[
            ('cat_encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), final_categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline by chaining all steps
    preprocessing_pipeline = Pipeline(steps=[
        ('route_creator', RouteTransformer()),
        ('urgency_creator', BookingUrgencyTransformer()),
        ('encoder', encoder)
    ])

    return preprocessing_pipeline

# This part is for demonstration if you run the file directly
if __name__ == '__main__':
    # Load raw data, dropping the original index column
    df_raw = pd.read_csv('flight_price.csv').drop('Unnamed: 0', axis=1)

    # Separate features and target
    X = df_raw.drop('price', axis=1)
    y = df_raw['price']

    # Create the full preprocessing pipeline
    pipeline = create_preprocessing_pipeline()

    # Fit the pipeline on the feature data and transform it
    X_processed = pipeline.fit_transform(X)

    print("Preprocessing pipeline executed successfully.")
    print(f"Original feature shape: {X.shape}")
    # Note: The processed output is a NumPy array, so we get its shape directly.
    print(f"Processed feature shape: {X_processed.shape}")
    
    # To see the column names after transformation (optional)
    # Get feature names from the one-hot encoder step
    ohe_feature_names = pipeline.named_steps['encoder'].named_transformers_['cat_encoder'].get_feature_names_out()
    # Get remainder feature names (those that were passed through)
    remainder_feature_names = [col for col in X.columns if col not in ['airline', 'departure_time', 'stops', 'arrival_time', 'class', 'source_city', 'destination_city', 'days_left', 'flight']]
    
    # Combine them
    all_feature_names = list(ohe_feature_names) + remainder_feature_names
    # print("\nFeature names after processing:")
    # print(all_feature_names)


'''
*Note: I have simplified the `create_preprocessor` function to be more direct. A more advanced version could use `sklearn.pipeline.Pipeline` to chain the custom transformers and the `ColumnTransformer` together.*

### 2. What to Keep in `__init__.py`

For your project, you can simply keep the `__init__.py` file **empty**.

Its purpose is to tell Python that the directory is a "package," which allows you to do clean imports from other files. For example, because you have an `__init__.py` file in the same directory, your `train.py` script can now import your preprocessor like this:

```python
from preprocessing import create_preprocessor
```

### 3. Preprocessing in Your Model Code (`train.py`)

No, you should **not** repeat the preprocessing steps in your model code. The entire point of the `preprocessing.py` file is to centralize that logic.

Your `train.py` script will be the *user* of your preprocessing module. Here is the correct workflow for `train.py`:

1.  **Import necessary libraries and your preprocessor.**
2.  **Load the raw data** (`flight_price.csv`).
3.  **Perform Feature Engineering:** Apply the custom transformers from `preprocessing.py` to your raw data.
4.  **Separate Target Variable:** Create your `X` (features) and `y` (target) DataFrames. Apply the log transformation to `y`.
5.  **Split Data:** Split `X` and `y` into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`).
6.  **Fit and Transform:**
    * Create the preprocessor: `preprocessor = create_preprocessor()`.
    * **Fit it ONLY on the training data:** `preprocessor.fit(X_train)`.
    * **Transform both sets:**
        * `X_train_processed = preprocessor.transform(X_train)`
        * `X_test_processed = preprocessor.transform(X_test)`
7.  **Train your model** on `X_train_processed` and `y_train`.
8.  **IMPORTANT: Save two files:**
    * The trained model (e.g., `model.pkl`).
    * The **fitted preprocessor object** (e.g., `preprocessor.pkl`). You will need this in your FastAPI app to transform live, incoming data in exactly the same w

'''