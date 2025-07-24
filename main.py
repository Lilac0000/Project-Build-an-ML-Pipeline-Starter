import os
import hydra
from omegaconf import DictConfig
import json
import mlflow
import tempfile

# Define the steps of the pipeline
_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]

@hydra.main(config_path=".", config_name="config", version_base="1.3")
def go(config: DictConfig):
    # Set environment variables for Weights & Biases tracking
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
    
    # Determine which steps to run
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        # Step 1: Download dataset from artifact hub or URL
        if "download" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version="main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )
        
        # Step 2: Basic Cleaning
        if "basic_cleaning" in active_steps:
            mlflow.run(
                os.path.abspath("src/basic_cleaning"),
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact": f"{config['main']['entity']}/{config['main']['project_name']}/sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_data",
                    "output_description": "Cleaned data",
                    "min_price": float(config["etl"]["min_price"]),
                    "max_price": float(config["etl"]["max_price"]),
                },
            )
        
        # Step 3: Data Quality Check
        if "data_check" in active_steps:
            mlflow.run(
                os.path.abspath("src/data_check"),
                "main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": float(config["data_check"]["min_price"]),
                    "max_price": float(config["data_check"]["max_price"]),
                },
            )
        
        # Step 4: Train/Validation/Test Split
        if "data_split" in active_steps:
            mlflow.run(
                os.path.abspath("src/data_split"),
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",
                    "test_size": float(config["etl"]["test_size"]),
                    "random_state": int(config["etl"]["random_seed"]),
                    "stratify": config["etl"]["stratify_col"],
                },
            )
        
        # Step 5: Train Random Forest Model
        if "train_random_forest" in active_steps:
            rf_config_path = os.path.join(tmp_dir, "rf_config.json")
            with open(rf_config_path, "w") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)
            
            mlflow.run(
                os.path.abspath("src/train_random_forest"),
                "main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval.csv:latest",
                    "random_forest_config": rf_config_path,
                    "val_size": float(config["modeling"]["val_size"]),
                    "stratify": config["modeling"]["stratify"],
                },
            )

if __name__ == "__main__":
    go()
