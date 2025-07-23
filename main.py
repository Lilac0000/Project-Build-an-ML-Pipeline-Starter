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
]

@hydra.main(config_name="config")
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config.main.project_name
    os.environ["WANDB_RUN_GROUP"] = config.main.experiment_name

    steps_param = config.main.steps
    active_steps = steps_param.split(",") if steps_param != "all" else _steps

    for step in active_steps:
        print(f"Running step: {step}")

        if step == "train_random_forest":
            rf_config_path = os.path.abspath("rf_config.json")
            with open(rf_config_path, "w") as f:
                json.dump(dict(config.modeling.random_forest.items()), f)

            mlflow.run(
                uri=".",
                entry_point=step,
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval.csv:latest",
                    "random_forest_config": rf_config_path,
                    "val_size": float(config.modeling.val_size),
                    "stratify": config.modeling.stratify,
                },
            )
        elif step == "download":
            mlflow.run(
                uri=".",
                entry_point=step,
                env_manager="conda",
                parameters={
                    "sample": config.etl.sample,
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )
        elif step == "basic_cleaning":
            mlflow.run(
                uri=".",
                entry_point=step,
                env_manager="conda",
                parameters={
                    "input_artifact": f"{config.main.entity}/{config.main.project_name}/sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_data",
                    "output_description": "Cleaned data",
                    "min_price": float(config.etl.min_price),
                    "max_price": float(config.etl.max_price),
                },
            )
        elif step == "data_check":
            mlflow.run(
                uri=".",
                entry_point=step,
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "min_price": float(config.etl.min_price),
                    "max_price": float(config.etl.max_price),
                    "kl_threshold": float(config.data_check.kl_threshold),
                },
            )
        elif step == "data_split":
            mlflow.run(
                uri=".",
                entry_point=step,
                env_manager="conda",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",
                    "test_size": float(config.etl.test_size),
                    "random_state": int(config.etl.random_seed),
                    "stratify": config.etl.stratify_col,
                },
            )
        else:
            print(f"Warning: step {step} is not recognized")

if __name__ == "__main__":
    go()
