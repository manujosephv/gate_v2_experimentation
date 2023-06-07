import os
import sys
from src.generate_dataset_pipeline import generate_dataset
import traceback  # Needed for pulling out your full stackframe info
from src.train import *
import wandb
import platform
import time
import torch
import numpy as np
from pathlib import Path

#os.environ["WANDB_MODE"] = "offline"

def modify_config(config):
    for key in config.keys():
        if key.endswith("_temp"):
            new_key = "model__" + key[:-5]
            print("Replacing value from key", key, "to", new_key)
            if config[key] == "None":
                config[new_key] = None
            else:
                config[new_key] = config[key]
     
    return config

DATA_PATH = Path("data")

def download_save_data(config=None):
    print("GPU?")
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    #    print(torch.cuda.current_device())
    #    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print("#####")
    CONFIG_DEFAULT = {"train_prop": 0.70,
                      "val_test_prop": 0.3,
                      "max_val_samples": 50000,
                      "max_test_samples": 50000}
    # "model__use_checkpoints": True} #TODO

    config.update(CONFIG_DEFAULT)
    print(config)
    # Modify the config in certain cases
    config = modify_config(config)
    
    # print(config)
    DATA_PATH.mkdir(exist_ok=True)
    save_path = DATA_PATH.joinpath(config['dataset_name'])
    save_path.mkdir(exist_ok=True)
    # try:
    if config["n_iter"] == "auto":
        x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, np.random.RandomState(0))
        if x_test.shape[0] > 6000:
            n_iter = 1
        elif x_test.shape[0] > 3000:
            n_iter = 2
        elif x_test.shape[0] > 1000:
            n_iter = 3
        else:
            n_iter = 5
    else:
        n_iter = config["n_iter"]
        
    for i in range(n_iter):
        rng = np.random.RandomState(i)
        print(rng.randn(1))
        t = time.time()
        x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, rng)
        data_generation_time = time.time() - t
        print("Data generation time:", data_generation_time)
        # print(y_train)
        print(x_train.shape)

        if config["regression"]:
            y_train, y_val, y_test = y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(
                np.float32)
        else:
            y_train, y_val, y_test = y_train.reshape(-1), y_val.reshape(-1), y_test.reshape(-1)
            # y_train, y_val, y_test = y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32)
        x_train, x_val, x_test = x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(
            np.float32)
        np.save(save_path/f"x_train_fold_{i}.npy", x_train)
        np.save(save_path/f"x_val_fold_{i}.npy", x_val)
        np.save(save_path/f"x_test_fold_{i}.npy", x_test)
        np.save(save_path/f"y_train_fold_{i}.npy", y_train)
        np.save(save_path/f"y_val_fold_{i}.npy", y_val)
        np.save(save_path/f"y_test_fold_{i}.npy", y_test)
        np.save(save_path/f"categorical_indicator_fold_{i}.npy", categorical_indicator)
        np.save(save_path/f"config_fold_{i}.npy", config)

            

    # except:
    #     # Print to the console
    #     print("ERROR")
    #     # To get the traceback information
    #     print(traceback.format_exc())
    #     print(config)
    #     return -1
    # return 0


if __name__ == """__main__""":
    # config = {'data__categorical': False, 'data__keyword': 'sulfur', 'data__method_name': 'real_data', 'data__regression': True,
    #  'max_train_samples': 10000, 'model__criterion': 'squared_error', 'model__learning_rate': 0.0096842939564663,
    #  'model__loss': 'huber', 'model__max_depth': 5, 'model__max_leaf_nodes': 5, 'model__min_impurity_decrease': 0,
    #  'model__min_samples_leaf': 25, 'model__min_samples_split': 2, 'model__n_estimators': 1000,
    #  'model__n_iter_no_change': 20, 'model__subsample': 0.9976203905983656, 'model__validation_fraction': 0.2,
    #  'model_name': 'gbt_r', 'model_type': 'sklearn', 'n_iter': 'auto', 'one_hot_encoder': True, 'regression': True,
    #  'transformed_target': False, 'train_prop': 0.7, 'val_test_prop': 0.3, 'max_val_samples': 50000,
    #  'max_test_samples': 50000}

    # config = {"model_type": "sklearn",
    #           "model_name": "gbt_c",
    #           "regression": False,
    #           "data__regression": False,
    #           "data__categorical": False,
    #           "model__n_estimators": 100,
    #           "data__method_name": "real_data",
    #           "data__keyword": "california",
    #           "n_iter": "auto",
    #           "wandb": False}

    config = {
        'wandb': False,

        'max_test_samples': 50000.0,
        'max_train_samples': 10000.0,
        'max_val_samples': 50000.0,
        'train_prop': 0.7,
        'val_test_prop': 0.3,
        'n_iter': 'auto',
        'data__keyword': '44129',
        'dataset_name': 'Higgs',        
        'regression': 0.0,
        'data__regression': 0.0,
        'data__categorical': 0.0,
        # 'use_gpu': nan,
        # 'early_stopping_rounds': nan,
        # 'transform__0__apply_on': nan,
        # 'transform__0__method_name': nan,
        # 'transform__0__type': nan,
        }
    # config = {
    #     "model_type": "skorch",
    #     "model_name": "npt",
    #     "n_iter": 1,
    #     "model__optimizer": "adamw",
    #     "model__lr": 0.001,
    #     "model__batch_size": 64,
    #     "data__method_name": "real_data",
    #     "data__keyword": "electricity",
    #     "regression": False
    # }

    # config = {"model_type": "skorch",
    #           "model_name": "rtdl_resnet",
    #           "regression": False,
    #           "data__regression": False,
    #           "data__categorical": False,
    #           "n_iter": 1,
    #           "max_train_samples": 1000,
    #           "model__device": "cpu",
    #           "model__optimizer": "adam",
    #           "model__lr_scheduler": "adam",
    #           "model__use_checkpoints": True,
    #           "model__batch_size": 64,
    #           "model__max_epochs": 10,
    #           "model__lr": 1e-3,
    #           "model__module__n_layers": 2,
    #           "model__module__d": 64,
    #           "model__module__d_hidden_factor": 3,
    #           "model__module__hidden_dropout": 0.2,
    #           "model__module__residual_dropout": 0.1,
    #           "model__module__d_embedding": 64,
    #           "model__module__normalization": "batchnorm",
    #           "model__module__activation": "reglu",
    #           "data__method_name": "real_data",
    #           "data__keyword": "electricity",
    #           #"max_train_samples": None,
    #           #"max_test_samples": None,
    #           }

    download_save_data(config)