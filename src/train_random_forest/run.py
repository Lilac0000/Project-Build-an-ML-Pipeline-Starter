#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder
import wandb


def delta_date_feature(dates):
    """
    Given a 2d array with dates (in a format that can be parsed by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (max(d) - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed

    ######################################
    # Use run.use_artifact(...).file() to get the train and validation artifact (args.trainval_artifact)
    # and save the returned path in the trainval_local_path variable
    # YOUR CODE HERE
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    ######################################

    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")  # this removes the column "price" from X and puts it into y

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(args, rf_config)

    ######################################
    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
    # YOUR CODE HERE
    sk_pipe.fit(X_train, y_train)
    ######################################

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    ######################################
    # Log the r_squared and the mae to W&B. Use run.summary[...] = ...
    # YOUR CODE HERE
    run.summary["r2"] = r_squared
    run.summary["mae"] = mae
    ######################################

    # Upload to W&B the feture importance coming from the random forest model
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    ######################################
    # Save the figure of the feature importance and log it to W&B. 
    # Call the figure "feature_importance.png"
    # YOUR CODE HERE
    fig_feat_imp.savefig("feature_importance.png")
    run.log_artifact("feature_importance.png")
    ######################################

    # Save model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    ######################################
    # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory "random_forest_dir"
    # and use mlflow.sklearn.save_model for that
    # YOUR CODE HERE
    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=mlflow.models.infer_signature(X_val, y_pred)
    )
    ######################################

    ######################################
    # Upload the model we just exported to W&B
    # Use run.log_artifact. The artifact should be called args.output_artifact, should be of type
    # "model_export", have a description of "Trained Random Forest" and be the "random_forest_dir" directory
    # YOUR CODE HERE
    run.log_artifact(
        "random_forest_dir",
        name=args.output_artifact,
        type="model_export",
        description="Trained Random Forest"
    )
    ######################################

    # Plot a histogram of the residuals
    fig_residuals = plot_residuals(sk_pipe, X_val, y_val)
    fig_residuals.savefig("residuals.png")
    run.log_artifact("residuals.png")


def plot_feature_importance(pipe, feat_names):

    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)-1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx])
    sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    sub_feat_imp.set_xticklabels(np.array(feat_names)[idx])
    plt.setp(sub_feat_imp.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    sub_feat_imp.set_ylim([0, feat_imp.max() * 1.1])
    sub_feat_imp.set_xlabel("Features")
    sub_feat_imp.set_ylabel("Importance")
    sub_feat_imp.set_title("Feature importance")
    fig_feat_imp.tight_layout()

    return fig_feat_imp


def plot_residuals(pipe, X_val, y_val):
    # Calculate residuals
    y_pred = pipe.predict(X_val)
    residuals = y_val - y_pred

    fig_residuals, sub_residuals = plt.subplots(figsize=(10, 10))
    sub_residuals.hist(residuals, bins=50)
    sub_residuals.set_xlabel("Residuals")
    sub_residuals.set_ylabel("Frequency")
    sub_residuals.set_title("Histogram of residuals")
    fig_residuals.tight_layout()

    return fig_residuals


def get_inference_pipeline(args, rf_config):

    # Let's define the steps in the feature engineering pipeline. 
    # We will fill in the details later.
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    ordinal_categorical_preproc = OrdinalEncoder()

    ######################################
    # Build a preprocessing pipeline for ordinal categorical features
    # This should 1) fill in missing values with the most frequent value
    # and 2) apply an OrdinalEncoder step. 
    # Use the SimpleImputer class, and use the most_frequent strategy
    # Call the pipeline ordinal_categorical_preproc
    # YOUR CODE HERE
    ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder()
    )
    ######################################

    ######################################
    # Build a preprocessing pipeline for non-ordinal categorical features
    # This should 1) fill in missing values with the constant string "missing"
    # and 2) apply a OrdinalEncoder step
    # Call the pipeline non_ordinal_categorical_preproc
    # YOUR CODE HERE 
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder()
    )
    ######################################

    # Let's impute the numerical columns to make sure we can handle missing values
    # We use the median strategy, but you could try other strategies (mean, most_frequent, etc.)
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # Arbitrary imputer of 1 for the number of days column
    date_imputer = SimpleImputer(strategy="constant", fill_value=1)

    # Build the pipeline as a list of steps
    steps = [
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
                ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
                ("impute_zero", zero_imputer, zero_imputed),
                ("impute_date", date_imputer, ["days_since"]),
                ("tfidf", TfidfVectorizer(
                    max_features=args.max_tfidf_features,
                    stop_words='english'
                ), "name")
            ],
            remainder="drop"  # This drops the columns that we do not transform
        )),
        ("random_forest", RandomForestRegressor(**rf_config))
    ]

    sk_pipe = Pipeline(steps)

    # The list of the feature names is useful for later plotting and debugging
    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["days_since", "name"]
    
    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact", 
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size", 
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", 
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False
    )

    parser.add_argument(
        "--stratify_by", 
        type=str,
        help="Column to use for stratification",
        default='none',
        required=False
    )

    parser.add_argument(
        "--rf_config", 
        help="Random forest configuration. A YAML string.",
        required=True
    )

    parser.add_argument(
        "--max_tfidf_features", 
        help="Maximum number of words to be used in the TFIDF vectorizer",
        type=int,
        default=10
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output serialized model",
        required=True
    )

    args = parser.parse_args()

    go(args)
