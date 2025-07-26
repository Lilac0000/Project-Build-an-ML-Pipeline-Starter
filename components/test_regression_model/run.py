#!/usr/bin/env python
"""
This script tests a trained Random Forest model
"""
import argparse
import logging
import pandas as pd
import numpy as np
import wandb
import mlflow.sklearn
import tempfile
import os
from sklearn.metrics import mean_absolute_error

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading model artifact from W&B...")
    model_local_path = run.use_artifact(args.mlflow_model).download()

    logger.info(f"Files in model artifact: {os.listdir(model_local_path)}")

    logger.info("Downloading test dataset...")
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    logger.info("Reading test dataset...")
    X_test = pd.read_csv(test_dataset_path)

    logger.info("Loading model and extracting feature information...")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    
    # Log what we loaded
    logger.info(f"Loaded object type: {type(sk_pipe)}")
    
    # If the model is a dictionary containing the pipeline and other info
    if isinstance(sk_pipe, dict):
        model_data = sk_pipe
        logger.info(f"Loaded object keys: {model_data.keys()}")
        
        # Extract the actual pipeline
        sk_pipe = model_data['model']
        expected_features = model_data.get('numeric_features', []) + model_data.get('categorical_features', [])
        
        logger.info(f"Expected features: {expected_features}")
        logger.info(f"Test dataset columns before filtering: {list(X_test.columns)}")
        
        # Remove extra features not used in training
        extra_features = [col for col in X_test.columns if col not in expected_features]
        if extra_features:
            logger.info(f"Removing extra features from test data: {extra_features}")
            X_test = X_test[expected_features]
        
        logger.info(f"Using features: {list(X_test.columns)}")
        
        # Apply label encoders to categorical features
        if 'categorical_features' in model_data and 'label_encoders' in model_data:
            for feature in model_data['categorical_features']:
                if feature in X_test.columns and feature in model_data['label_encoders']:
                    le = model_data['label_encoders'][feature]
                    logger.info(f"Encoding categorical feature: {feature}")
                    
                    # Handle unseen categories by mapping them to the first class
                    def safe_encode(x):
                        if x in le.classes_:
                            return le.transform([x])[0]
                        else:
                            logger.warning(f"Unseen category '{x}' in feature '{feature}', using default")
                            return le.transform([le.classes_[0]])[0]
                    
                    X_test[feature] = X_test[feature].apply(safe_encode)
        
        # Ensure feature order matches training
        if expected_features:
            X_test = X_test[expected_features]
    
    else:
        # If it's just the pipeline directly
        logger.info("Model is a direct sklearn pipeline")
        
    logger.info(f"Final model type: {type(sk_pipe)}")
    logger.info(f"Test data shape after preprocessing: {X_test.shape}")
    logger.info(f"Test data dtypes:\n{X_test.dtypes}")
    
    # Check for any remaining string columns
    string_cols = X_test.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        logger.error(f"Found string columns that weren't encoded: {list(string_cols)}")
        logger.error(f"Sample values: {X_test[string_cols].head()}")
        raise ValueError(f"String columns found: {list(string_cols)}")
    
    logger.info("Performing inference...")
    y_pred = sk_pipe.predict(X_test)

    logger.info(f"Predictions shape: {y_pred.shape}")
    logger.info(f"Sample predictions: {y_pred[:5]}")

    # If we have ground truth labels, calculate MAE
    if 'price' in pd.read_csv(test_dataset_path).columns:
        y_true = pd.read_csv(test_dataset_path)['price']
        mae = mean_absolute_error(y_true, y_pred)
        logger.info(f"Mean Absolute Error: {mae}")
        
        # Log to W&B
        run.summary["mae"] = mae
    
    logger.info("Test completed successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test a trained model")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
