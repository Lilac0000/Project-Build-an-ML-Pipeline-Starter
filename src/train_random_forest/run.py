#!/usr/bin/env python
"""
This step trains a random forest model using the provided training data,
evaluates it, and logs parameters, metrics, artifacts, and the model to MLflow.
"""

import argparse
import logging
import tempfile
import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_data, split_data
from src.preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def train_model(X_train, y_train, args):
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    metrics = {
        "rmse_train": mean_squared_error(y_train, y_train_pred, squared=False),
        "mae_train": mean_absolute_error(y_train, y_train_pred),
        "r2_train": r2_score(y_train, y_train_pred),
        "rmse_val": mean_squared_error(y_val, y_val_pred, squared=False),
        "mae_val": mean_absolute_error(y_val, y_val_pred),
        "r2_val": r2_score(y_val, y_val_pred),
    }
    return metrics, y_val_pred


def create_visualizations(model, X_train, y_val, y_val_pred, feature_names):
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")

    # Residuals plot
    residuals = y_val - y_val_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("residuals.png")

    return importances


def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest model.")
    parser.add_argument("--input_artifact", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set size")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of trees")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Min samples required to split")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Min samples at a leaf node")

    args = parser.parse_args()

    logger.info("Loading data...")
    df = load_data(args.input_artifact)

    logger.info("Splitting data...")
    train_set, val_set = split_data(df, val_size=args.val_size, random_seed=args.random_seed)

    logger.info("Preprocessing data...")
    X_train, y_train, X_val, y_val, label_encoders, X = preprocess_data(train_set, val_set)

    logger.info("Starting MLflow run...")
    with mlflow.start_run(nested=True):
        logger.info("Training model...")
        model = train_model(X_train, y_train, args)

        logger.info("Evaluating model...")
        metrics, y_val_pred = evaluate_model(model, X_train, y_train, X_val, y_val)

        logger.info("Logging parameters...")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("random_seed", args.random_seed)
        mlflow.log_param("val_size", args.val_size)

        logger.info("Logging metrics...")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info("Logging artifacts...")
        create_visualizations(model, X_train, y_val, y_val_pred, X.columns)
        mlflow.log_artifact('feature_importance.png')
        mlflow.log_artifact('residuals.png')

        logger.info("Saving model and logging it...")
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "random_forest_model.pkl")
            model_data = {
                'model': model,
                'label_encoders': label_encoders,
                'feature_names': list(X.columns)
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
