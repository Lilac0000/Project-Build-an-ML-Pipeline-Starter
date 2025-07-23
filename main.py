import json
import mlflow
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
    # "test_regression_model"  # Uncomment if needed
]

@hydra.main(config_name='config')
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_param = config['main']['steps']
    active_steps = steps_param.split(",") if steps_param != "all" else _steps

    # Since subfolders might not be full MLflow projects, run steps in this root project
    for step in active_steps:
        print(f"Running step: {step}")
        if step == "train_random_forest":
            # Write RF config to file
            rf_config_path = os.path.abspath("rf_config.json")
            with open(rf_config_path, "w") as f:
                json.dump(dict(config["modeling"]["random_forest"].items()), f)

            mlflow.run(
                uri=".",
                entry_point=step,
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval.csv:latest",
                    "random_forest_config": rf_config_path,
                    "val_size": float(config["modeling"]["val_size"]),
                    "stratify": config["modeling"]["stratify"],
                },
            )
        else:
            mlflow.run(
                uri=".",
                entry_point=step,
                env_manager="conda",
                parameters={
                    "sample": config["etl"].get("sample", ""),
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                    "input_artifact": f"{config['main']['entity']}/{config['main']['project_name']}/sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_data",
                    "output_description": "Cleaned data",
                    "min_price": float(config["etl"].get("min_price", 0)),
                    "max_price": float(config["etl"].get("max_price", 0)),
                    "kl_threshold": float(config["data_check"].get("kl_threshold", 0)),
                    "test_size": float(config["etl"].get("test_size", 0)),
                    "random_state": int(config["etl"].get("random_seed", 42)),
                    "stratify": config["etl"].get("stratify_col", None),
                    "mlflow_model": "random_forest_export:prod",
                    "test_data": "test.csv:latest",
                },
            )


if __name__ == "__main__":
    go()
