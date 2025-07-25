import argparse
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import mlflow
import wandb


def plot_residuals(model, X, y):
    y_pred = model.predict(X)
    residuals = y - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, bins=50, kde=True, ax=ax)
    ax.set_title("Residuals Histogram")
    ax.set_xlabel("Residuals")
    return fig


def main(args):
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    mlflow.set_experiment("RandomForestRegression")

    # Load artifact
    artifact_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_path)
    
    print("Training Random Forest model...")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Split data
    stratify_col = df[args.stratify_by] if args.stratify_by.lower() != "none" else None
    train, val = train_test_split(
        df, test_size=args.val_size, stratify=stratify_col, random_state=args.random_seed
    )

    # Prepare features and target
    # Select only numeric columns and some categorical ones for the model
    numeric_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
                       'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
    categorical_features = ['neighbourhood_group', 'room_type']
    
    # Filter columns that actually exist in the dataset
    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    # Handle missing values in reviews_per_month (fill with 0)
    train['reviews_per_month'] = train['reviews_per_month'].fillna(0)
    val['reviews_per_month'] = val['reviews_per_month'].fillna(0)
    
    # Prepare training data
    X_train_numeric = train[numeric_features]
    X_val_numeric = val[numeric_features]
    
    # Encode categorical features
    X_train_cat = train[categorical_features].copy()
    X_val_cat = val[categorical_features].copy()
    
    # Simple label encoding for categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train_cat[col] = le.fit_transform(X_train_cat[col].astype(str))
        X_val_cat[col] = le.transform(X_val_cat[col].astype(str))
        label_encoders[col] = le
    
    # Combine features
    X_train = pd.concat([X_train_numeric, X_train_cat], axis=1)
    X_val = pd.concat([X_val_numeric, X_val_cat], axis=1)
    
    y_train = train[args.target]
    y_val = val[args.target]
    
    print(f"Features used: {X_train.columns.tolist()}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")

    # Pipeline with only StandardScaler (now all features are numeric)
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

    # Train
    with mlflow.start_run():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        print(f"Model performance - R2: {r2:.4f}, MAE: {mae:.4f}")

        # Log to MLflow
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        
        # Log to W&B (both summary and as metrics)
        run.summary["r2"] = r2
        run.summary["mae"] = mae
        run.log({"r2": r2, "mae": mae})
        run.log({"n_estimators": args.n_estimators, "max_depth": args.max_depth})

        # Feature importance
        feat_importances = pipe.named_steps["rf"].feature_importances_
        feat_imp_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": feat_importances
        }).sort_values("importance", ascending=False)

        print("Top 5 most important features:")
        print(feat_imp_df.head())

        fig_feat = plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=feat_imp_df)
        plt.title("Feature Importances")
        plt.tight_layout()
        fig_feat.savefig("feature_importance.png")

        # Log feature importance as artifact
        feat_artifact = wandb.Artifact(
            "feature_importance", type="image", description="Feature importance plot"
        )
        feat_artifact.add_file("feature_importance.png")
        run.log_artifact(feat_artifact)

        # Save model and preprocessors
        os.makedirs("random_forest_dir", exist_ok=True)
        model_export = {
            "model": pipe,
            "label_encoders": label_encoders,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features
        }
        
        with open("random_forest_dir/model.pkl", "wb") as f:
            pickle.dump(model_export, f)
        
        print("Model saved successfully!")

        # Log model artifact
        model_artifact = wandb.Artifact(
            args.output_artifact,
            type="model_export",
            description="Trained Random Forest model with preprocessors"
        )
        model_artifact.add_dir("random_forest_dir")
        run.log_artifact(model_artifact)

        # Residuals plot
        fig_resid = plot_residuals(pipe, X_val, y_val)
        fig_resid.savefig("residuals.png")

        resid_artifact = wandb.Artifact(
            "residuals", type="image", description="Model residuals plot"
        )
        resid_artifact.add_file("residuals.png")
        run.log_artifact(resid_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model.")

    parser.add_argument("--input_artifact", type=str, required=True, help="Input data artifact")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set size")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stratify_by", type=str, default="none", help="Column to stratify by")

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)

    parser.add_argument("--output_artifact", type=str, required=True, help="Output model artifact name")
    parser.add_argument("--target", type=str, required=True, help="Target column name")

    args = parser.parse_args()
    main(args)
