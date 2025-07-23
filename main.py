import os
import json
import tempfile
import hydra
from omegaconf import DictConfig

# Import your step modules (make sure these modules exist and have a run() function)
from src import get_data, basic_cleaning, data_check, data_split, train_random_forest, test_regression_model


_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model",
]

@hydra.main(config_name="config")
def go(config: DictConfig):

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            print("Running download step...")
            get_data.run(
                sample=config["etl"]["sample"],
                artifact_name="sample.csv",
                artifact_type="raw_data",
                artifact_description="Raw file as downloaded",
                tmp_dir=tmp_dir
            )

        if "basic_cleaning" in active_steps:
            print("Running basic_cleaning step...")
            basic_cleaning.run(
                input_artifact=f"{config['main']['entity']}/{config['main']['project_name']}/sample.csv:latest",
                output_artifact="clean_sample.csv",
                output_type="clean_data",
                output_description="Cleaned data",
                min_price=float(config["etl"]["min_price"]),
                max_price=float(config["etl"]["max_price"]),
                tmp_dir=tmp_dir
            )

        if "data_check" in active_steps:
            print("Running data_check step...")
            data_check.run(
                csv="clean_sample.csv:latest",
                ref="clean_sample.csv:reference",
                min_price=float(config["etl"]["min_price"]),
                max_price=float(config["etl"]["max_price"]),
                kl_threshold=float(config["data_check"]["kl_threshold"]),
                tmp_dir=tmp_dir
            )

        if "data_split" in active_steps:
            print("Running data_split step...")
            data_split.run(
                input_artifact="clean_sample.csv:latest",
                test_size=float(config["etl"]["test_size"]),
                random_state=int(config["etl"]["random_seed"]),
                stratify=config["etl"]["stratify_col"],
                tmp_dir=tmp_dir
            )

        if "train_random_forest" in active_steps:
            print("Running train_random_forest step...")
            rf_config_path = os.path.join(tmp_dir, "rf_config.json")
            with open(rf_config_path, "w") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            train_random_forest.run(
                trainval_artifact="trainval.csv:latest",
                random_forest_config=rf_config_path,
                val_size=float(config["modeling"]["val_size"]),
                stratify=config["modeling"]["stratify"],
                tmp_dir=tmp_dir
            )

        if "test_regression_model" in active_steps:
            print("Running test_regression_model step...")
            test_regression_model.run(
                mlflow_model="random_forest_export:prod",
                test_data="test.csv:latest",
                tmp_dir=tmp_dir
            )


if __name__ == "__main__":
    go()
