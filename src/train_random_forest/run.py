import argparse
import os
import pickle
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_residuals(model, X, y):
    """Plot residuals histogram for model evaluation."""
    y_pred = model.predict(X)
    residuals = y - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, bins=50, kde=True, ax=ax)
    ax.set_title("Residuals Histogram")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    return fig


def plot_predictions_vs_actual(model, X, y):
    """Plot predictions vs actual values."""
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predictions vs Actual Values")
    return fig


def validate_inputs(df, args):
    """Validate input data and arguments."""
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset columns: {df.columns.tolist()}")
    
    if args.stratify_by.lower() != "none" and args.stratify_by not in df.columns:
        raise ValueError(f"Stratify column '{args.stratify_by}' not found in dataset")
    
    if args.val_size <= 0 or args.val_size >= 1:
        raise ValueError("Validation size must be between 0 and 1")
    
    logger.info(f"Input validation passed. Dataset shape: {df.shape}")


def prepare_features(df, numeric_features, categorical_features):
    """Prepare and validate features exist in dataset."""
    # Filter only existing columns
    existing_numeric = [col for col in numeric_features if col in df.columns]
    existing_categorical = [col for col in categorical_features if col in df.columns]
    
    missing_numeric = set(numeric_features) - set(existing_numeric)
    missing_categorical = set(categorical_features) - set(existing_categorical)
    
    if missing_numeric:
        logger.warning(f"Missing numeric features: {missing_numeric}")
    if missing_categorical:
        logger.warning(f"Missing categorical features: {missing_categorical}")
    
    logger.info(f"Using numeric features: {existing_numeric}")
    logger.info(f"Using categorical features: {existing_categorical}")
    
    return existing_numeric, existing_categorical


def encode_categorical_features(X_train_cat, X_val_cat, categorical_features):
    """Encode categorical features with proper handling of unseen categories."""
    label_encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        
        # Combine train and validation to handle unseen categories
        combined_cats = pd.concat([
            X_train_cat[col].astype(str), 
            X_val_cat[col].astype(str)
        ]).unique()
        
        le.fit(combined_cats)
        
        # Transform both sets
        X_train_cat[col] = le.transform(X_train_cat[col].astype(str))
        X_val_cat[col] = le.transform(X_val_cat[col].astype(str))
        
        label_encoders[col] = le
        logger.info(f"Encoded {col}: {len(le.classes_)} unique categories")
    
    return label_encoders


def main(args):
    try:
        # End any active MLflow run to avoid conflicts
        if mlflow.active_run() is not None:
            logger.info("Ending existing MLflow run")
            mlflow.end_run()

        # Initialize W&B run
        run = wandb.init(job_type="train_random_forest")
        run.config.update(vars(args))

        # Set MLflow experiment
        mlflow.set_experiment("RandomForestRegression")

        # Load input artifact data
        logger.info(f"Loading artifact: {args.input_artifact}")
        artifact_path = run.use_artifact(args.input_artifact).file()
        df = pd.read_csv(artifact_path)

        logger.info("Training Random Forest model...")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Validate inputs
        validate_inputs(df, args)

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
        with mlflow.start_run():
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
            mlflow.log_metrics({
                "train_r2": train_r2,
                "val_r2": val_r2,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse
            })

            # Log parameters to MLflow
            mlflow.log_params({
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
            mlflow.log_artifact("feature_importance.png")

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
            mlflow.log_artifact("residuals.png")

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
            mlflow.log_artifact("predictions_vs_actual.png")

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
            mlflow.sklearn.log_model(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest model for regression tasks."
    )

    # Data arguments
    parser.add_argument(
        "--input_artifact", 
        type=str, 
        required=True, 
        help="Input data artifact name"
    )
    parser.add_argument(
        "--target", 
        type=str, 
        required=True, 
        help="Target column name for regression"
    )
    parser.add_argument(
        "--output_artifact", 
        type=str, 
        required=True, 
        help="Output model artifact name"
    )

    # Split arguments
    parser.add_argument(
        "--val_size", 
        type=float, 
        default=0.2, 
        help="Validation set size (0.0 to 1.0)"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--stratify_by", 
        type=str, 
        default="none", 
        help="Column to stratify by (or 'none')"
    )

    # Model hyperparameters
    parser.add_argument(
        "--n_estimators", 
        type=int, 
        default=100, 
        help="Number of trees in the forest"
    )
    parser.add_argument(
        "--max_depth", 
        type=int, 
        default=10, 
        help="Maximum depth of trees (0 for unlimited)"
    )
    parser.add_argument(
        "--min_samples_split", 
        type=int, 
        default=2, 
        help="Minimum samples required to split a node"
    )
    parser.add_argument(
        "--min_samples_leaf", 
        type=int, 
        default=1, 
        help="Minimum samples required at a leaf node"
    )

    args = parser.parse_args()
    main(args)
