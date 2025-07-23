import os
import subprocess
import hydra
from omegaconf import DictConfig
import pathlib
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

def data_check_step():
    print("Running data checks with pytest...")
    test_path = pathlib.Path(__file__).parent / "src" / "data_check" / "test_data.py"
    result = subprocess.run(["pytest", str(test_path), "-v"])
    if result.returncode != 0:
        print("Data checks failed!")
        exit(1)
    else:
        print("Data checks passed!")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
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
            data_check_step()

        if "data_split" in active_steps:
            data_split_path = os.path.abspath("src/data_split")
            mlflow.run(
                data_split_path,
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

            train_rf_path = os.path.abspath("src/train_random_forest")
            mlflow.run(
                train_rf_path,
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval.csv:latest",
                    "random_forest_config": rf_config,
                    "val_size": float(config["modeling"]["val_size"]),
                    "stratify": config["modeling"]["stratify"],
                },
            )

if __name__ == "__main__":
    go()
