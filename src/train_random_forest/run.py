#!/usr/bin/env python
"""
This script trains a Random Forest model with comprehensive error handling and logging.
Fixed to work within MLflow pipeline context without creating conflicting runs.
"""
import argparse
import logging
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_input_data(df):
    """Validate input dataframe."""
    logger.info(f"Input validation passed. Dataset shape: {df.shape}")
    
    required_columns = ['price', 'neighbourhood_group', 'room_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for target variable
    if df['price'].isnull().sum() > 0:
        logger.warning(f"Found {df['price'].isnull().sum()} missing values in target variable")
    
    return True

def prepare_features(df):
    """Prepare features for training."""
    # Select numeric features (excluding target and id columns)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and id columns
    numeric_features = [col for col in numeric_features if col not in ['price', 'id']]
    
    logger.info(f"Using numeric features: {numeric_features}")
    
    # Select categorical features
    categorical_features = ['neighbourhood_group', 'room_type']
    logger.info(f"Using categorical features: {categorical_features}")
    
    # Prepare feature matrix
    X = df[numeric_features].copy()
    
    # Encode categorical features
    label_encoders = {}
    for cat_feature in categorical_features:
        if cat_feature in df.columns:
            le = LabelEncoder()
            X[cat_feature] = le.fit_transform(df[cat_feature].astype(str))
            label_encoders[cat_feature] = le
            logger.info(f"Encoded {cat_feature}: {len(le.classes_)} unique categories")
    
    # Get target
    y = df['price'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    logger.info(f"Features used: {list(X.columns)}")
    
    return X, y, label_encoders

def train_model(X_train, y_train, args):
    """Train Random Forest model."""
    rf_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'random_state': args.random_seed,
        'n_jobs': -1
    }
    
    logger.info(f"Training Random Forest with parameters: {rf_params}")
    
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    
    return rf

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluate model and return metrics."""
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Validation predictions
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }
    
    logger.info(f"Training RMSE: {train_rmse:.4f}")
    logger.info(f"Validation RMSE: {val_rmse:.4f}")
    logger.info(f"Validation R²: {val_r2:.4f}")
    
    return metrics, y_val_pred

def create_visualizations(model, X_train, y_val, y_val_pred, feature_names):
    """Create and save visualizations."""
    # Feature importance plot
    plt.figure(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Residuals plot
    plt.figure(figsize=(10, 6))
    residuals = y_val - y_val_pred
    plt.scatter(y_val_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_importance

def main(args):
    """Main training function."""
    try:
        # Initialize wandb
        run = wandb.init(
            project="Project-Build-an-ML-Pipeline-Starter-src_basic_cleaning",
            job_type="train_random_forest"
        )
        
        logger.info(f"Loading artifact: {args.input_artifact}")
        
        # Download artifact
        artifact = run.use_artifact(args.input_artifact)
        artifact_path = artifact.file()
        
        # Load data
        df = pd.read_csv(artifact_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Validate input
        validate_input_data(df)
        
        # Prepare features
        X, y, label_encoders = prepare_features(df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=args.val_size, 
            random_state=args.random_seed,
            stratify=df[args.stratify_by] if args.stratify_by in df.columns else None
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
        
        logger.info("Training Random Forest model...")
        
        # **KEY FIX: Don't create a new MLflow run - use the existing one**
        # The original code had: with mlflow.start_run():
        # This caused the conflict. Instead, log to the current active run.
        
        # Train model
        model = train_model(X_train, y_train, args)
        
        # Evaluate model
        metrics, y_val_pred = evaluate_model(model, X_train, y_train, X_val, y_val)
        
        # Log parameters to MLflow
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("random_seed", args.random_seed)
        mlflow.log_param("val_size", args.val_size)
        
        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Create visualizations
        feature_importance = create_visualizations(model, X_train, y_val, y_val_pred, X.columns)
        
        # Log artifacts to MLflow
        mlflow.log_artifact('feature_importance.png')
        mlflow.log_artifact('residuals.png')
        
        # Save and log model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "random_forest_model.pkl")
            
            # Save model with label encoders
            model_data = {
                'model': model,
                'label_encoders': label_encoders,
                'feature_names': list(X.columns)
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                model, 
                "model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
            )
            
            # Upload model as wandb artifact
            model_artifact = wandb.Artifact(
                args.output_artifact,
                type="model",
                description="Trained Random Forest model with preprocessing"
            )
            model_artifact.add_file(model_path)
            run.log_artifact(model_artifact)
        
        # Log metrics to wandb
        wandb.log(metrics)
        wandb.log({
            "feature_importance_plot": wandb.Image("feature_importance.png"),
            "residuals_plot": wandb.Image("residuals.png")
        })
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved as artifact: {args.output_artifact}")
        
        # Clean up visualization files
        for file in ['feature_importance.png', 'residuals.png']:
            if os.path.exists(file):
                os.remove(file)
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e
    
    finally:
        # Always finish wandb run
        if 'run' in locals():
            run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )
    
    parser.add_argument(
        "--val_size", 
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
        default=0.2
    )
    
    parser.add_argument(
        "--random_seed", 
        type=int,
        help="Seed for random number generator",
        default=42
    )
    
    parser.add_argument(
        "--stratify_by", 
        type=str,
        help="Column to use for stratification",
        default="neighbourhood_group"
    )
    
    parser.add_argument(
        "--n_estimators",
        type=int,
        help="Number of estimators for Random Forest",
        default=100
    )
    
    parser.add_argument(
        "--max_depth",
        type=int,
        help="Maximum depth of trees",
        default=10
    )
    
    parser.add_argument(
        "--min_samples_split",
        type=int,
        help="Minimum samples required to split an internal node",
        default=4
    )
    
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        help="Minimum samples required to be at a leaf node",
        default=3
    )
    
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact",
        default="random_forest_export"
    )
    
    parser.add_argument(
        "--target",
        type=str,
        help="Target column name",
        default="price"
    )
    
    args = parser.parse_args()
    
    main(args)
