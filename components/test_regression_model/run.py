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

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading model artifact from W&B...")
    model_artifact = run.use_artifact(args.mlflow_model, type='model')
    model_local_dir = model_artifact.download()

    # 🟢 The exported MLflow model is usually in a subfolder like 'model'
    model_path = os.path.join(model_local_dir, "model")

    logger.info("Downloading test dataset...")
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    logger.info("Reading test dataset...")
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("price")

    logger.info("Loading model and performing inference...")
    sk_pipe = mlflow.sklearn.load_model(model_path)
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
