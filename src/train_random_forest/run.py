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
    X_val = val.drop(columns=[args.targ]()_
