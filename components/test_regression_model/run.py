#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
import os
import joblib
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading model artifact from W&B...")
    model_artifact = run.use_artifact(args.mlflow_model, type='model_export')
    model_local_dir = model_artifact.download()
    
    # Check what files are in the artifact directory
    files_in_artifact = os.listdir(model_local_dir)
    logger.info(f"Files in model artifact: {files_in_artifact}")
    
    # Look for the model file directly in the artifact directory
    model_file_path = os.path.join(model_local_dir, "model.pkl")
    
    logger.info("Downloading test dataset...")
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    logger.info("Reading test dataset...")
    X_test = pd.read_csv(test_dataset_path)
    
    # Extract the target column before preprocessing
    y_test = X_test.pop("price")
    
    logger.info("Loading model and extracting feature information...")
    # Load the model directly using joblib since it's a pickle file
    loaded_object = joblib.load(model_file_path)
    
    # Debug: check what type of object was loaded
    logger.info(f"Loaded object type: {type(loaded_object)}")
    logger.info(f"Loaded object keys: {loaded_object.keys() if isinstance(loaded_object, dict) else 'Not a dict'}")
    
    # If it's a dictionary, extract the actual model and feature info
    if isinstance(loaded_object, dict):
        sk_pipe = loaded_object['model']
        numeric_features = loaded_object.get('numeric_features', [])
        categorical_features = loaded_object.get('categorical_features', [])
        
        # Filter the test data to only include the features the model was trained on
        expected_features = numeric_features + categorical_features
        logger.info(f"Expected features: {expected_features}")
        logger.info(f"Test dataset columns before filtering: {list(X_test.columns)}")
        
        # Keep only the columns that were used for training
        missing_features = [f for f in expected_features if f not in X_test.columns]
        extra_features = [f for f in X_test.columns if f not in expected_features]
        
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")
        if extra_features:
            logger.info(f"Removing extra features from test data: {extra_features}")
        
        # Select only the features used during training
        available_features = [f for f in expected_features if f in X_test.columns]
        X_test = X_test[available_features]
        logger.info(f"Using features: {available_features}")
        
    else:
        sk_pipe = loaded_object
    
    logger.info(f"Final model type: {type(sk_pipe)}")
    logger.info(f"Test data shape after preprocessing: {X_test.shape}")
    
    logger.info("Performing inference...")
    y_pred = sk_pipe.predict(X_test)

    logger.info("Scoring model...")
    r_squared = sk_pipe.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"R² Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    run.summary['r2'] = r_squared
    run.summary['mae'] = mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")
    
    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model artifact (e.g. 'your-entity/project/model_export:prod')",
        required=True
    )
    
    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset artifact (e.g. 'test_data.csv:latest')",
        required=True
    )

    args = parser.parse_args()
    go(args)
