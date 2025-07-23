import os
import json
import hydra
from omegaconf import DictConfig

def download_data(config):
    print("Downloading data...")
    # Your download logic here

def basic_cleaning(config):
    print("Cleaning data...")
    # Your cleaning logic here

def data_check(config):
    print("Checking data...")
    # Your data check logic here

def data_split(config):
    print("Splitting data...")
    # Your splitting logic here

def train_random_forest(config):
    print("Training model...")
    # Your training logic here

@hydra.main(config_name="config")
def main(config: DictConfig):
    steps = config.main.steps.split(",") if config.main.steps != "all" else [
        "download", "basic_cleaning", "data_check", "data_split", "train_random_forest"
    ]

    if "download" in steps:
        download_data(config)
    if "basic_cleaning" in steps:
        basic_cleaning(config)
    if "data_check" in steps:
        data_check(config)
    if "data_split" in steps:
        data_split(config)
    if "train_random_forest" in steps:
        train_random_forest(config)

if __name__ == "__main__":
    main()
