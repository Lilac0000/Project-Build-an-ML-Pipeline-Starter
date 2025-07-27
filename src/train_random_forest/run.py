#!/usr/bin/env python
"""
This step trains a random forest model using the provided training data,
evaluates it, and logs parameters, metrics, artifacts, and the model to MLflow.
"""
import sys
import os  # <-- added missing import

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import logging
import tempfile
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

def safe_mlflow_log(log_func, *args, **kwargs):
    """Safely log to MLflow only if there is an active run."""
    try:
        if mlflow.active_run():
            return log_func(*args, **kwargs)
        else:
            logger.warning("No active MLflow run - skipping MLflow logging")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


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
    plt.close()

    # Residuals plot
    residuals = y_val - y_val_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("residuals.png")
    plt.close()

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
    with mlflow.start_run():
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
        mlflow.log_artifact("feature_importance.png")
        mlflow.log_artifact("residuals.png")

<<<<<<< HEAD
        # Stratify column or None
        stratify_col = df[args.stratify_by] if args.stratify_by.lower() != "none" else None

        # Split train/val
        train, val = train_test_split(
            df, 
            test_size=args.val_size, 
            stratify=stratify_col, 
            random_state=args.random_seed
        )

        # Define features to use
        numeric_features = [
            'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
            'reviews_per_month', 'calculated_host_listings_count', 'availability_365'
        ]
        categorical_features = ['neighbourhood_group', 'room_type']

        # Prepare features (filter existing columns)
        numeric_features, categorical_features = prepare_features(
            df, numeric_features, categorical_features
        )

        # Create copies to avoid SettingWithCopyWarning
        train = train.copy()
        val = val.copy()

        # Handle missing values
        if 'reviews_per_month' in numeric_features:
            train['reviews_per_month'] = train['reviews_per_month'].fillna(0)
            val['reviews_per_month'] = val['reviews_per_month'].fillna(0)

        # Prepare features
        X_train_numeric = train[numeric_features]
        X_val_numeric = val[numeric_features]

        X_train_cat = train[categorical_features].copy()
        X_val_cat = val[categorical_features].copy()

        # Encode categorical features
        label_encoders = encode_categorical_features(
            X_train_cat, X_val_cat, categorical_features
        )

        # Combine features
        X_train = pd.concat([X_train_numeric, X_train_cat], axis=1)
        X_val = pd.concat([X_val_numeric, X_val_cat], axis=1)

        y_train = train[args.target]
        y_val = val[args.target]

        logger.info(f"Features used: {X_train.columns.tolist()}")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")

        # Build pipeline: scaler + random forest
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth if args.max_depth > 0 else None,
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_seed,
                n_jobs=-1
            ))
        ])

        # Start MLflow run context
        # MLflow run context already active - removed conflicting start_run
        logger.info("Training model...")
        
        # Train model
        pipe.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = pipe.predict(X_train)
        y_pred_val = pipe.predict(X_val)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
        val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)

        logger.info(f"Training metrics - R2: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
        logger.info(f"Validation metrics - R2: {val_r2:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

        # Log metrics to MLflow
        safe_mlflow_log(mlflow.log_metrics, {
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse
        })

        # Log parameters to MLflow
        safe_mlflow_log(mlflow.log_params, {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "val_size": args.val_size,
            "random_seed": args.random_seed,
            "stratify_by": args.stratify_by
        })

        # Log to W&B
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

        # Feature importance analysis
        feat_importances = pipe.named_steps["rf"].feature_importances_
        feat_imp_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": feat_importances
        }).sort_values("importance", ascending=False)

        logger.info("Top 5 most important features:")
        logger.info(feat_imp_df.head().to_string(index=False))

        # Create feature importance plot
        fig_feat = plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp_df.head(10), x="importance", y="feature")
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        
        # Save and log feature importance plot
        fig_feat.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
        safe_mlflow_log(mlflow.log_artifact, "feature_importance.png")

        feat_artifact = wandb.Artifact(
            "feature_importance", 
            type="image", 
            description="Feature importance plot"
        )
        feat_artifact.add_file("feature_importance.png")
        run.log_artifact(feat_artifact)

        # Create and save residuals plot
        fig_resid = plot_residuals(pipe, X_val, y_val)
        fig_resid.savefig("residuals.png", dpi=150, bbox_inches='tight')
        safe_mlflow_log(mlflow.log_artifact, "residuals.png")

        resid_artifact = wandb.Artifact(
            "residuals", 
            type="image", 
            description="Model residuals plot"
        )
        resid_artifact.add_file("residuals.png")
        run.log_artifact(resid_artifact)

        # Create predictions vs actual plot
        fig_pred = plot_predictions_vs_actual(pipe, X_val, y_val)
        fig_pred.savefig("predictions_vs_actual.png", dpi=150, bbox_inches='tight')
        safe_mlflow_log(mlflow.log_artifact, "predictions_vs_actual.png")

        pred_artifact = wandb.Artifact(
            "predictions_vs_actual", 
            type="image", 
            description="Predictions vs actual values plot"
        )
        pred_artifact.add_file("predictions_vs_actual.png")
        run.log_artifact(pred_artifact)

        # Prepare model export
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

        # Save model
        model_path = "random_forest_dir/model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_export, f)

        logger.info(f"Model saved to {model_path}")

        # Log model to MLflow
        safe_mlflow_log(mlflow.sklearn.log_model, 
            pipe, 
            "random_forest_model",
            registered_model_name="RandomForestRegressor"
        )

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
        if 'run' in locals():
            run.finish(exit_code=1)
        raise e

    finally:
        # Ensure W&B run is finished
        if 'run' in locals():
            run.finish()
        
        # Ensure MLflow run is ended
        if mlflow.active_run():
            mlflow.end_run()
=======
        logger.info("Saving model and logging it...")
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "random_forest_model.pkl")
            model_data = {
                "model": model,
                "label_encoders": label_encoders,
                "feature_names": list(X.columns),
            }
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            mlflow.log_artifact(model_path)
            mlflow.sklearn.log_model(model, "model")
>>>>>>> 28f79337f0c363b906adae24bd814dcd3e71a068


if __name__ == "__main__":
    main()
