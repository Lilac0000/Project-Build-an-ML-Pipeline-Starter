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
from sklearn.preprocessing import StandardScaler
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

    # Load artifact
    artifact_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_path)

    # Split data
    stratify_col = df[args.stratify_by] if args.stratify_by.lower() != "none" else None
    train, val = train_test_split(
        df, test_size=args.val_size, stratify=stratify_col, random_state=args.random_seed
    )

    X_train = train.drop(columns=[args.target])
    y_train = train[args.target]
    X_val = val.drop(columns=[args.target])
    y_val = val[args.target]

    # Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_seed
        ))
    ])

    # Train model
    print("Training Random Forest model...")
    pipe.fit(X_train, y_train)
    print("Training completed!")

    # Make predictions
    y_pred = pipe.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # Log metrics to Wandb
    run.summary["r2"] = r2
    run.summary["mae"] = mae
    run.log({"r2": r2, "mae": mae})
    
    # Log hyperparameters
    run.log({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "val_size": args.val_size,
        "random_seed": args.random_seed
    })

    # Feature importance
    feat_importances = pipe.named_steps["rf"].feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": feat_importances
    }).sort_values("importance", ascending=False)

    print("Top 10 Most Important Features:")
    print(feat_imp_df.head(10))

    # Create feature importance plot
    fig_feat = plt.figure(figsize=(10, 8))
    top_features = feat_imp_df.head(15)  # Show top 15 features
    sns.barplot(x="importance", y="feature", data=top_features)
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    fig_feat.savefig("feature_importance.png", dpi=150, bbox_inches='tight')

    # Log feature importance as artifact
    feat_artifact = wandb.Artifact(
        "feature_importance", type="image", description="Feature importance plot"
    )
    feat_artifact.add_file("feature_importance.png")
    run.log_artifact(feat_artifact)

    # Create residuals plot
    fig_resid = plot_residuals(pipe, X_val, y_val)
    fig_resid.savefig("residuals.png", dpi=150, bbox_inches='tight')

    resid_artifact = wandb.Artifact(
        "residuals", type="image", description="Model residuals plot"
    )
    resid_artifact.add_file("residuals.png")
    run.log_artifact(resid_artifact)

    # Save model
    print("Saving model...")
    os.makedirs("random_forest_dir", exist_ok=True)
    with open("random_forest_dir/model.pkl", "wb") as f:
        pickle.dump(pipe, f)

    # Save feature names for later use
    feature_names = list(X_train.columns)
    with open("random_forest_dir/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    # Log model artifact to wandb
    model_artifac
