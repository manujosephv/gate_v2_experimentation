import uuid
from pathlib import Path
import requests

import joblib
import numpy as np
import torch
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import GatedAdditiveTreeEnsembleConfig

# from src.train import *
from src.tuning import OptunaTuner

# os.environ["WANDB_MODE"] = "offline"
DATA_DIR = Path("data")
TRAINING_DONE_TOKEN = "6035411225:AAFKDDN5-mU6NUN3uzo2Mls3M1gmvOTGpS0" #"6069721026:AAG5mDHXLhRKjFOwNrhsz0KMxTus9S9uV24"
TELEGRAM_CHAT_ID = 1006088451 #790388072

"""
config = {
    "dataset": "compas-two-years",
    "n_trials": 100,
    "batch_size": 512,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "optimizer": "AdamW",
    "strategy": "tpe",
    "pruning": False,
}
"""


def clean_up_folder(folder):
    # check if folder is str then make it Path
    if isinstance(folder, str):
        folder = Path(folder)
    # check if folder exists
    if not folder.exists():
        print(f"{folder} does not exist")
        return
    for f in folder.glob("*"):
        if f.is_dir():
            clean_up_folder(f)
        else:
            f.unlink()

def get_search_space(trial):
    space = {
        "gflu_stages": trial.suggest_int("gflu_stages", 2, 30),
        "gflu_dropout": trial.suggest_float("gflu_dropout", 0.0, 0.5),
        "num_trees": 0,
        # "feature_mask_function": trial.suggest_categorical(
        #     "feature_mask_function", ["entmax", "sparsemax", "softmax", "t-softmax"]
        # ),
        "feature_mask_function": "t-softmax",
        "optimizer_config__weight_decay": trial.suggest_categorical(
            "optimizer_config__weight_decay", [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        ),
        "head_config": {"layers": "32-16"},
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [1e-3, 1e-4, 1e-5, 1e-6]
        ),
    }
    if space["feature_mask_function"] == "t-softmax":
        space["gflu_feature_init_sparsity"] = trial.suggest_float(
            "gflu_feature_init_sparsity", 0.0, 0.9
        )
        space["learnable_sparsity"] = trial.suggest_categorical(
            "learnable_sparsity", [True, False]
        )
    return space


def notify_telegram(message):
    try:
        requests.get(
            f"https://api.telegram.org/bot{TRAINING_DONE_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}&parse_mode=html"
        )
    except Exception as e:
        raise e


def tune_model(config):
    print("GPU?")
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    #    print(torch.cuda.current_device())
    #    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print("#####")
    data_path = DATA_DIR / config["dataset"]
    d_config_files = list(data_path.glob("*config*"))
    d_config = np.load(d_config_files[0], allow_pickle=True).item()
    task = "classification" if d_config["regression"] == 0 else "regression"
    n_folds = len(d_config_files)
    if d_config.get("data__categorical",0) == 1:
        categorical_indicator = np.load(
            data_path / "categorical_indicator_fold_0.npy", allow_pickle=True
        )
        feat_names = [f"feature_{i}" for i in range(categorical_indicator.shape[0])]
        cat_col_names = [
            feat_names[i]
            for i in range(len(feat_names))
            if categorical_indicator[i] == 1
        ]
        num_col_names = [
            feat_names[i]
            for i in range(len(feat_names))
            if categorical_indicator[i] == 0
        ]
    else:
        n_features = np.load(
            data_path / "x_test_fold_0.npy", allow_pickle=True
        ).shape[1]
        cat_col_names = None
        num_col_names = [f"feature_{i}" for i in range(n_features)]
    print(f"Using Data from: {data_path} for task: {task} with {n_folds} folds")
    notify_telegram(
        f"Start Tuning for data: <b>{config['dataset']}</b> for task: {task} with <b>{n_folds}</b> folds."
    )
    data_config = DataConfig(
        target=["target"],
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names or [],
        normalize_continuous_features=True,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=False,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        early_stopping="valid_loss",
        early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
        early_stopping_patience=config[
            "early_stopping_patience"
        ],  # No. of epochs of degradation training will wait before terminating
        checkpoints="valid_loss",
        load_best=True,  # After training, load the best checkpoint
        progress_bar="none",  # Turning off Progress bar
        trainer_kwargs=dict(enable_model_summary=False),  # Turning off model summary
    )
    optimizer_config = OptimizerConfig(
        optimizer=config["optimizer"], optimizer_params={"weight_decay": 1e-5}
    )
    model_config = GatedAdditiveTreeEnsembleConfig(
        task=task,
        learning_rate=1e-3,
        metrics=["mean_squared_error"] if task == "regression" else None,
        metrics_prob_input=[False] if task == "regression" else None,
    )
    model_id = "940f259aa03a46dd8c92aab7e3d40035"
    study_name = f"{config['dataset']}_{model_id}"
    tuner = OptunaTuner(
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
        model_config=model_config,
        data_config=data_config,
        data_path=data_path,
        direction="minimize",
        metric_name="test_mean_squared_error" if task == "regression" else "test_accuracy",
        search_space_fn=get_search_space,
        study_name=study_name,
        storage=f"sqlite:///study/{study_name}.db",
        strategy=config["strategy"],
        pruning=config["pruning"],
        notify_progress=True,
    )
    joblib.dump(config, f"study/config_{study_name}.pkl")

    best_params, best_value, study_df = tuner.tune(
        n_trials=config["n_trials"], n_jobs=1
    )
    print(f"Best Parameters: {best_params}")
    print(f"Best Value: {best_value}")
    notify_telegram(
        f"Tuning complete for data from: <b>{data_path}</b> for task: <b>{task}</b> with <b>{n_folds}</b> folds."
        f"Best Value: <b>{best_value:.5f}</b>\n\nBest Parameters: {best_params}\nStudy Name: {study_name}"
    )
    study_df.to_csv(f"output/study_{config['dataset']}_{model_id}.csv")
    clean_up_folder("saved_models")


if __name__ == "__main__":
    # config = {
    #     "dataset": "Brazilian_houses",
    #     "n_trials": 100,
    #     "batch_size": 512,
    #     "max_epochs": 100,
    #     "early_stopping_patience": 10,
    #     "optimizer": "AdamW",
    #     "strategy": "tpe",
    #     "pruning": False,
    # }
    # tune_model(config)

    # config = {
    #     "dataset": "wine_quality",
    #     "n_trials": 100,
    #     "batch_size": 512,
    #     "max_epochs": 100,
    #     "early_stopping_patience": 10,
    #     "optimizer": "AdamW",
    #     "strategy": "tpe",
    #     "pruning": False,
    # }

    # tune_model(config)

    # config = {
    #     "dataset": "sulfur",
    #     "n_trials": 100,
    #     "batch_size": 512,
    #     "max_epochs": 100,
    #     "early_stopping_patience": 10,
    #     "optimizer": "AdamW",
    #     "strategy": "tpe",
    #     "pruning": False,
    # }
    # tune_model(config)

    # config = {
    #     "dataset": "visualizing_soil",
    #     "n_trials": 100,
    #     "batch_size": 512,
    #     "max_epochs": 100,
    #     "early_stopping_patience": 10,
    #     "optimizer": "AdamW",
    #     "strategy": "tpe",
    #     "pruning": False,
    # }
    # tune_model(config)

    # config = {
    #     "dataset": "medical_charges",
    #     "n_trials": 100,
    #     "batch_size": 512,
    #     "max_epochs": 100,
    #     "early_stopping_patience": 10,
    #     "optimizer": "AdamW",
    #     "strategy": "tpe",
    #     "pruning": False,
    # }
    # tune_model(config)

    config = {
        "dataset": "california_housing",
        "n_trials": 100,
        "batch_size": 512,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "optimizer": "AdamW",
        "strategy": "tpe",
        "pruning": False,
    }
    tune_model(config)