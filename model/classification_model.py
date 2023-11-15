from itertools import count
from typing import Any, Dict, Literal, Union

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from numpy import ndarray
from numpy.random import RandomState
from pandas import DataFrame
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.tree import DecisionTreeClassifier

import hyperopt
from hyperopt import hp, fmin, tpe
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from copy import deepcopy

from sklearn import metrics

import numpy as np


class TrainError(Exception):
    "if models are nor fitted"


class AdaBoostClf(AdaBoostClassifier):
    def __init__(self, max_depth: int = 4, *, n_estimators: int = 50, learning_rate: float = 1, algorithm: Literal['SAMME', 'SAMME.R'] = "SAMME.R", random_state: int | RandomState | None = None, base_estimator: Any = "deprecated") -> None:
        self.max_depth = max_depth
        super().__init__(DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators,
                         learning_rate=learning_rate, algorithm=algorithm, random_state=random_state, base_estimator=base_estimator)


"""
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

    balance_dict = {"ROS": RandomOverSampler(),
                    "SMOTE": SMOTE(),
                    "ADASYN": ADASYN(),
                    "None": None}
                    
    h_space_balance = {"ROS": (hp.uniform("shrinkage", 0, 2),
                                hp.uniform("sampling_strategy", 0.3, 1)),
                        "SMOTE": (hp.quniform("k_neighbors", 1, 11, 1),
                                  hp.uniform("sampling_strategy", 0.3, 1)),
                        "ADASYN": (hp.quniform("n_neighbors", 1, 11, 1),
                                   hp.uniform("sampling_strategy", 0.3, 1))}
"""


class ClassifierModel:
    clf_dict = {"CatBoost": CatBoostClassifier(random_seed=42, silent=True),
                "LGM": LGBMClassifier(random_seed=42),
                "XGB": XGBClassifier(),
                "RF": RandomForestClassifier(random_state=42),
                "AdaBoost": AdaBoostClf(random_state=42)
                }

    h_space_clf = {"CatBoost": {"depth": hp.uniformint("depth", 2, 10),
                                "n_estimators": hp.uniformint("n_estimators", 40, 100),
                                "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
                                "l2_leaf_reg": hp.uniform("l2_leaf_reg", 0.01, 10),
                                },

                   "LGM":      {"max_depth": hp.choice('max_depth', np.arange(2, 10+1, dtype=int)),
                                "n_estimators":  hp.choice('n_estimators', np.arange(40, 100+1, dtype=int)),
                                "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
                                "reg_lambda": hp.uniform("reg_lambda", 0.01, 1),
                                "subsample": hp.uniform("subsample", 1e-5, 1)
                                },

                   "XGB":      {"max_depth": hp.choice('max_depth', list(range(2, 10))),
                                "n_estimators":  hp.choice('n_estimators', np.arange(40, 100+1, dtype=int)),
                                "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
                                "reg_lambda": hp.uniform("reg_lambda", 0.01, 1),
                                },

                   "RF":       {"max_depth": hp.choice('max_depth', np.arange(10, 100+1, dtype=int)),
                                "n_estimators":  hp.choice('n_estimators', list(range(40, 100+1))),
                                },

                   "AdaBoost": {"max_depth": hp.choice('max_depth', list(range(2, 10+1))),
                                "n_estimators": hp.choice('n_estimators', list(range(40, 100+1))),
                                "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
                                }}

    def __init__(self) -> None:

        self.fitted = False
        self.models: Dict[str, object] = deepcopy(self.clf_dict)

    def fit(self, x: Union[ndarray, DataFrame], y: Union[ndarray, DataFrame]) -> None:
        """fit each classifier 

        Args:
            x (Union[ndarray, DataFrame]): x train
            y (Union[ndarray, DataFrame]): y train
        """
        for clf in self.models.values():
            clf.fit(x, y)
        self.fitted = True

    def predict(self, X_test: Union[ndarray, DataFrame]) -> dict[str, ndarray]:
        """predict for each classifier

        Args:
            X_test (Union[ndarray, DataFrame]): x for prediction

        Raises:
            TrainError: if models are not fitted 

        Returns:
            dict[str, ndarray]: prediction for each classifier
        """
        if not self.fitted:
            raise TrainError("isn't fitted yet")
        res = {name: self.models[name].predict(X_test) for name in self.models}
        return res

    def cv(self, X_train: DataFrame, y_train: DataFrame, time_per_clf: int = 10) -> dict[str, dict[str, int | float]]:
        """method for cross validation with StratifiedKFold:
            - split X_train, y_train into 5 folds
            - optimize every classifier in their oun hyperparameter space with f1 macro score function 
            - result optimal params for each classifier are best of folds (with best score) 

        Args:
            X_train (DataFrame): X values for cv
            y_train (DataFrame): y values for cv
            time_per_clf (int, optional): time of optimization one classifier on one split, result time: time_per_clf*25(folds*classification model). Defaults to 10.

        Returns:
            dict[str, dict[str, int | float]]: result of optimization
        """
        skf = StratifiedKFold(shuffle=True, random_state=42)

        x_cv_train, y_cv_train = None, None
        cv_res = {clf_name: {"score": 0} for clf_name in self.models}

        for clf_name in self.models:
            print(f"evaluate {clf_name}")
            for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
                x_cv_train, y_cv_train = X_train.iloc[train_index], y_train.iloc[train_index].values.ravel(
                )
                x_cv_test, y_cv_test = X_train.iloc[test_index], y_train.iloc[test_index].values.ravel(
                )
                clf_model = self.models[clf_name]
                balance = {"test": {},
                           "train": {}}
                for value, count in zip(*np.unique(y_cv_test, return_counts=True)):
                    balance["test"][str(value)] = count

                for value, count in zip(*np.unique(y_cv_train, return_counts=True)):
                    balance["train"][str(value)] = count

                best = fmin(
                                fn=lambda params: hyperopt_objective(params, clf_model, x_cv_train, y_cv_train, x_cv_test, y_cv_test),
                                space=self.h_space_clf[clf_name],
                                algo=tpe.suggest,
                                timeout=time_per_clf,
                                return_argmin=False
                            )

                score = -hyperopt_objective(best, clf_model, x_cv_train, y_cv_train, x_cv_test, y_cv_test)


                if score > cv_res[clf_name]["score"]:
                    cv_res[clf_name] = best
                    cv_res[clf_name]["score"] = score
                    cv_res[clf_name]["balance"] = balance

        return cv_res

    def set_params(self, params: dict[str, dict]) -> None:
        """set params for each classifier

        Args:
            params (dict[str, dict]): params to be setted
        """
        for name in self.models:
            self.models[name].set_params(**params[name])

def hyperopt_objective(params, model, x_train, y_train, x_test, y_test):
    _model = deepcopy(model).set_params(**params)
    _model.fit(x_train, y_train)
    y_pred = _model.predict(x_test)
    counts = {t[0]: t[1] for t in zip(*np.unique(y_test, return_counts=True))}
    metric_value = (metrics.f1_score(y_test, y_pred, pos_label=1)/counts[1] + metrics.f1_score(y_test, y_pred, pos_label=0)/counts[0]) / (1/counts[0] + 1/counts[1])
    return -metric_value


if __name__ == "__main__":
    c_model = ClassifierModel()
    x = DataFrame(np.random.random((100, 5)))
    y = DataFrame(np.random.randint(0, 2, 100))
    c_model.cv(x, y)
    
