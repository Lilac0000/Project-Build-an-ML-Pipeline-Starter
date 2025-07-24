import os
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import mlflow
import tempfile

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]

@hydra.main(config_path=".", config_name="config", version_base="1.3")
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
    
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        
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
        
        if "basic_cleaning" in active_steps:
            basic_cleaning_path = os.path.abspath("src/basic_cleaning")
            mlflow.run(
                basic_cleaning_path,
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
        
        if "data_check" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/data_check",
                "main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference", 
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["data_check"]["min_price"],
                    "max_price": config["data_check"]["max_price"]
                },
            )
        
        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                env_manager="conda",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": str(config["modeling"]["test_size"]),
                    "random_seed": str(config["modeling"]["random_seed"]),
                    "stratify_by": str(config["modeling"]["stratify_by"])
                },
            )
        
        if "train_random_forest" in active_steps:
            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = OmegaConf.to_yaml(config["modeling"]["random_forest"])
            
            train_rf_path = os.path.abspath("src/train_random_forest")
            _ = mlflow.run(
                train_rf_path,
                "main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export"
                },
            )

if __name__ == "__main__":
    go()
