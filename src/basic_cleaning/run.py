#!/usr/bin/env python
"""
This step performs basic data cleaning such as removing outliers
based on minimum and maximum price constraints.
"""

import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact
    logger.info("Downloading input artifact...")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    # Filter out outliers
    logger.info("Filtering data by price range...")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting 'last_review' to datetime...")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save cleaned data
    logger.info("Saving cleaned dataset...")
    df.to_csv("clean_sample.csv", index=False)

    # Upload cleaned dataset to W&B
    logger.info("Logging cleaned data to W&B as an artifact...")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    logger.info("Cleaned data logged to W&B.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of the data")

    parser.add_argument(
        "--input_artifact", 
        type=str, 
        help="Name of the input artifact to download from W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output artifact that will contain the cleaned data",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact, e.g., 'cleaned_data'",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float, 
        help="Minimum price to include in the cleaned dataset",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float, 
        help="Maximum price to include in the cleaned dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
