#!/usr/bin/env python
"""
This step trains a random forest model using the provided training data,
evaluates it, and logs parameters, metrics, artifacts, and the model to MLflow and W&B.
"""
import sys
import os
import argparse
import logging
import tempfile
import pickle

import mlflow
import mlflow.sklearn
import wandb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.preprocessing import prepare_features, encode_categorical_features
from src.visualization import plot_residuals, plot_predictions_vs_actual  # assume you have these helper functions

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def safe_mlflow_log(log_func, *args, **kwargs):
    """Safely log to MLflow only if there is an active run."""
    try:
        if mlflow.active_run():
            return log_func(*args, **kwargs)
        else:
            logger.warning("No active MLflow run - skipping MLflow logging")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest model.")
    parser.add_argument("--input_artifact", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set size")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of trees")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Min samples required to split")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Min samples at a leaf node")
    parser.add_argument("--stratify_by", type=str, default="none", help="Column to stratify by or 'none'")
    parser.add_argument("--target", type=str, default="price", help="Target column name")
    parser.add_argument("--output_artifact", type=str, default="random_forest_export", help="Output artifact name for W&B")

    args = parser.parse_args()

    logger.info("Loading data...")
    df = load_data(args.input_artifact)

    # Stratify column or None
    stratify_col = df[args.stratify_by] if args.stratify_by.lower() != "none" else None

    logger.info("Splitting train/validation sets...")
    train, val = train_test_split(
        df,
        test_size=args.val_size,
        stratify=stratify_col,
        random_state=args.random_seed
    )

    # Define features to use (customize as needed)
    numeric_features = [
        'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365'
    ]
    categorical_features = ['neighbourhood_group', 'room_type']

    logger.info("Preparing features...")
    numeric_features, categorical_features = prepare_features(
        df, numeric_features, categorical_features
    )

    # Create copies to avoid SettingWithCopyWarning
    train = train.copy()
    val = val.copy()

    # Handle missing values for numeric features (example)
    if 'reviews_per_month' in numeric_features:
        train['reviews_per_month'] = train['reviews_per_month'].fillna(0)
        val['reviews_per_month'] = val['reviews_per_month'].fillna(0)

    X_train_numeric = train[numeric_features]
    X_val_numeric = val[numeric_features]

    X_train_cat = train[categorical_features].copy()
    X_val_cat = val[categorical_features].copy()

    logger.info("Encoding categorical features...")
    label_encoders = encode_categorical_features(
        X_train_cat, X_val_cat, categorical_features
    )

    X_train = pd.concat([X_train_numeric, X_train_cat], axis=1)
    X_val = pd.concat([X_val_numeric, X_val_cat], axis=1)

    y_train = train[args.target]
    y_val = val[args.target]

    logger.info(f"Features used: {X_train.columns.tolist()}")
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Validation set shape: {X_val.shape}")

    # Initialize W&B run
    run = wandb.init(project="Project-Build-an-ML-Pipeline-Starter", job_type="train")

    try:
        # Build pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth if args.max_depth and args.max_depth > 0 else None,
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_seed,
                n_jobs=-1
            ))
        ])

        logger.info("Training model...")
        pipe.fit(X_train, y_train)

        logger.info("Making predictions...")
        y_pred_train = pipe.predict(X_train)
        y_pred_val = pipe.predict(X_val)

        logger.info("Calculating metrics...")
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
        val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)

        logger.info(f"Training metrics - R2: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
        logger.info(f"Validation metrics - R2: {val_r2:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

        # Log parameters & metrics to MLflow
        safe_mlflow_log(mlflow.log_params, {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "val_size": args.val_size,
            "random_seed": args.random_seed,
            "stratify_by": args.stratify_by
        })
        safe_mlflow_log(mlflow.log_metrics, {
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse
        })

        # Log metrics to W&B
        wandb_metrics = {
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse
        }
        run.summary.update(wandb_metrics)
        run.log(wandb_metrics)

        # Feature importance
        feat_importances = pipe.named_steps["rf"].feature_importances_
        feat_imp_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": feat_importances
        }).sort_values("importance", ascending=False)

        logger.info("Top 5 most important features:")
        logger.info(feat_imp_df.head().to_string(index=False))

        # Plot and save feature importance
        fig_feat = plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp_df.head(10), x="importance", y="feature")
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        fig_feat.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close(fig_feat)

        safe_mlflow_log(mlflow.log_artifact, "feature_importance.png")

        feat_artifact = wandb.Artifact(
            "feature_importance",
            type="image",
            description="Feature importance plot"
        )
        feat_artifact.add_file("feature_importance.png")
        run.log_artifact(feat_artifact)

        # Residuals plot
        fig_resid = plot_residuals(pipe, X_val, y_val)  # You must define this function in src/visualization.py
        fig_resid.savefig("residuals.png", dpi=150, bbox_inches='tight')
        plt.close(fig_resid)

        safe_mlflow_log(mlflow.log_artifact, "residuals.png")

        resid_artifact = wandb.Artifact(
            "residuals",
            type="image",
            description="Model residuals plot"
        )
        resid_artifact.add_file("residuals.png")
        run.log_artifact(resid_artifact)

        # Predictions vs actual plot
        fig_pred = plot_predictions_vs_actual(pipe, X_val, y_val)  # Define this function as well
        fig_pred.savefig("predictions_vs_actual.png", dpi=150, bbox_inches='tight')
        plt.close(fig_pred)

        safe_mlflow_log(mlflow.log_artifact, "predictions_vs_actual.png")

        pred_artifact = wandb.Artifact(
            "predictions_vs_actual",
            type="image",
            description="Predictions vs actual values plot"
        )
        pred_artifact.add_file("predictions_vs_actual.png")
        run.log_artifact(pred_artifact)

        # Prepare model export folder
        os.makedirs("random_forest_dir", exist_ok=True)

        model_export = {
            "model": pipe,
            "label_encoders": label_encoders,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_names": X_train.columns.tolist(),
            "target_column": args.target,
            "model_metrics": {
                "val_r2": val_r2,
                "val_mae": val_mae,
                "val_rmse": val_rmse
            }
        }

        # Save model pickle
        model_path = "random_forest_dir/model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_export, f)

        logger.info(f"Model saved to {model_path}")

        # Log model to MLflow
        safe_mlflow_log(mlflow.sklearn.log_model,
                        pipe,
                        "random_forest_model",
                        registered_model_name="RandomForestRegressor")

        # Log model artifact to W&B
        model_artifact = wandb.Artifact(
            args.output_artifact,
            type="model_export",
            description="Trained Random Forest model with preprocessors and metadata"
        )
        model_artifact.add_dir("random_forest_dir")
        run.log_artifact(model_artifact)

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        run.finish(exit_code=1)
        raise e

    finally:
        run.finish()
        if mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main()
