import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # "test_regression_model"  # Uncomment if you want to run this step explicitly
]

@hydra.main(config_name='config')
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            mlflow.run(
                "src/basic_cleaning",
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
                "src/data_check",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "min_price": float(config["etl"]["min_price"]),
                    "max_price": float(config["etl"]["max_price"]),
                    "kl_threshold": float(config["data_check"]["kl_threshold"]),
                },
            )

        if "data_split" in active_steps:
            mlflow.run(
                "src/data_split",
                env_manager="conda",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",
                    "test_size": float(config["etl"]["test_size"]),
                    "random_state": int(config["etl"]["random_seed"]),
                    "stratify": config["etl"]["stratify_col"],
                },
            )

        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            mlflow.run(
                "src/train_random_forest",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval.csv:latest",
                    "random_forest_config": rf_config,
                    "val_size": float(config["modeling"]["val_size"]),
                    "stratify": config["modeling"]["stratify"],
                },
            )

        if "test_regression_model" in active_steps:
            mlflow.run(
                "src/test_regression_model",
                env_manager="conda",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_data": "test.csv:latest",
                },
            )

if __name__ == "__main__":
    go()
