from copy import deepcopy

import numpy as np
import optuna
import pandas as pd
# from optuna.integration import PyTorchLightningPruningCallback
from pytorch_tabular import TabularModel


def get_search_space(trial):
    space = {
        "gflu_stages": trial.suggest_int("gflu_stages", 2, 10),
        "gflu_dropout": trial.suggest_float("gflu_dropout", 0.0, 0.5),
        "tree_depth": 0,
        "feature_mask_function": trial.suggest_categorical(
            "feature_mask_function", ["entmax", "sparsemax", "softmax", "t-softmax"]
        ),
        "optimizer_config__weight_decay": trial.suggest_categorical(
            "optimizer_config__weight_decay", [1e-3, 1e-4, 1e-5, 1e-6]
        ),
        "head_config": {"layers": "32-16"},
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [1e-3, 1e-4, 1e-5, 1e-6]
        ),
    }
    if space["activation"] == "t-softmax":
        space["gflu_feature_init_sparsity"] = trial.suggest_float(
            "gflu_feature_init_sparsity", 0.0, 0.9
        )
        space["learnable_sparsity"] = trial.suggest_categorical(
            "learnable_sparsity", [True, False]
        )
    return space


class OptunaTuner:
    def __init__(
        self,
        trainer_config,
        optimizer_config,
        model_config,
        data_config,
        data_path,
        direction,
        metric_name,
        search_space_fn,
        pruning=True,
        strategy="random",
        **kwargs,
    ):
        self.trainer_config = trainer_config
        self.optimizer_config = optimizer_config
        self.model_config = model_config
        self.data_config = data_config
        self.data_path = data_path
        self.search_space_fn = search_space_fn
        self.direction = direction
        self.metric_name = metric_name
        self.pruning = pruning
        pruner = (
            optuna.pruners.PercentilePruner(
                percentile=75.0, n_startup_trials=10, n_warmup_steps=5
            )
            if pruning
            else optuna.pruners.NopPruner()
        )
        # self._uuid = uuid.uuid4().hex

        kwargs["direction"] = direction
        kwargs["pruner"] = kwargs.get("pruner", pruner)
        kwargs["load_if_exists"] = True
        kwargs["sampler"] = (
            optuna.samplers.TPESampler(seed=42)
            if strategy == "tpe"
            else optuna.samplers.RandomSampler(seed=42)
        )
        self.optuna_study_kwargs = kwargs
        self.study = optuna.create_study(**self.optuna_study_kwargs)
        self.n_fold = len(list(data_path.glob("*config*")))

    def objective(self, trial):
        trainer_config_t = deepcopy(self.trainer_config)
        optimizer_config_t = deepcopy(self.optimizer_config)
        model_config_t = deepcopy(self.model_config)

        params = self.search_space_fn(trial)
        print(f"Trial Params: {params}")
        # Now set the parameters to the right config
        for name, param in params.items():
            if name.startswith("optimizer"):
                optimizer_config_t.optimizer_params["weight_decay"] = param
            else:
                setattr(model_config_t, name, param)

        metrics = []
        for fold in range(self.n_fold):
            x_train = np.load(self.data_path / f"x_train_fold_{fold}.npy", allow_pickle=True)
            y_train = np.load(self.data_path / f"y_train_fold_{fold}.npy", allow_pickle=True).reshape(
                -1, 1
            )
            x_val = np.load(self.data_path / f"x_val_fold_{fold}.npy", allow_pickle=True)
            y_val = np.load(self.data_path / f"y_val_fold_{fold}.npy", allow_pickle=True).reshape(-1, 1)
            x_test = np.load(self.data_path / f"x_train_fold_{fold}.npy", allow_pickle=True)
            y_test = np.load(self.data_path / f"y_train_fold_{fold}.npy", allow_pickle=True).reshape(
                -1, 1
            )
            # combine x and y into a dataframe
            train = pd.DataFrame(
                np.concatenate([x_train, y_train], axis=1),
                columns=[f"feature_{i}" for i in range(x_train.shape[1])]
                + ["target"],
            )
            val = pd.DataFrame(
                np.concatenate([x_val, y_val], axis=1),
                columns=[f"feature_{i}" for i in range(x_val.shape[1])] + ["target"],
            )
            test = pd.DataFrame(
                np.concatenate([x_test, y_test], axis=1),
                columns=[f"feature_{i}" for i in range(x_test.shape[1])] + ["target"],
            )
            # Initialize the tabular model
            tabular_model = TabularModel(
                data_config=self.data_config,
                model_config=model_config_t,
                optimizer_config=optimizer_config_t,
                trainer_config=trainer_config_t,
            )
            # try:
            # Fit the model
            tabular_model.fit(
                train=train,
                validation=val,
                # callbacks=[
                #     PyTorchLightningPruningCallback(trial, monitor="valid_loss")
                # ],
            )
            result = tabular_model.evaluate(test, verbose=False)
            metrics.append(result[0][self.metric_name])
            # except RuntimeError as e:
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     metrics.append(0 if self.direction == "maximize" else 1e6)
        ret_metric = np.nanmean(metrics)
        print("Trial Value: ", ret_metric)
        return ret_metric

    def tune(self, n_trials, timeout=None, n_jobs=1):
        self.study.optimize(
            self.objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs
        )
        # Return the best params, trials dataframe and best value
        return (
            self.study.best_params,
            self.study.best_value,
            self.study.trials_dataframe(),
        )
