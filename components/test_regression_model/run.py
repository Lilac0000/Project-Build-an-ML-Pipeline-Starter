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
    y_test = X_test.pop("price")

    logger.info("Loading model and performing inference...")
    # Load the model directly using joblib since it's a pickle file
    loaded_object = joblib.load(model_file_path)
    
    # Debug: check what type of object was loaded
    logger.info(f"Loaded object type: {type(loaded_object)}")
    logger.info(f"Loaded object keys (if dict): {loaded_object.keys() if isinstance(loaded_object, dict) else 'Not a dict'}")
    
    # If it's a dictionary, extract the actual model
    if isinstance(loaded_object, dict):
        # Common keys where models are stored
        if 'model' in loaded_object:
            sk_pipe = loaded_object['model']
        elif 'pipeline' in loaded_object:
            sk_pipe = loaded_object['pipeline']
        elif 'estimator' in loaded_object:
            sk_pipe = loaded_object['estimator']
        else:
            # Print all keys to see what's available
            logger.info(f"Available keys: {list(loaded_object.keys())}")
            # Try the first key that looks like it might contain a model
            possible_keys = [k for k in loaded_object.keys() if any(word in k.lower() for word in ['model', 'pipe', 'estimator', 'clf', 'regressor'])]
            if possible_keys:
                sk_pipe = loaded_object[possible_keys[0]]
                logger.info(f"Using key: {possible_keys[0]}")
            else:
                raise ValueError(f"Could not find model in dictionary. Available keys: {list(loaded_object.keys())}")
    else:
        sk_pipe = loaded_object
    
    logger.info(f"Final model type: {type(sk_pipe)}")
    
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
