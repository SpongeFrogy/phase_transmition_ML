from typing import Any, Dict, Literal, Union

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
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


class AdaBoostClf(AdaBoostClassifier):
    def __init__(self, max_depth: int = 4, *, n_estimators: int = 50, learning_rate: float = 1, algorithm: Literal['SAMME', 'SAMME.R'] = "SAMME.R", random_state: int | RandomState | None = None, base_estimator: Any = "deprecated") -> None:
        self.max_depth = max_depth
        super().__init__(DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators,
                         learning_rate=learning_rate, algorithm=algorithm, random_state=random_state, base_estimator=base_estimator)


"""
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
                                "reg_lambda": hp.uniform("reg_lambda", 0.01, 10),
                                },

                   "XGB":      {"max_depth": hp.choice('max_depth', list(range(40, 100))),
                                "n_estimators":  hp.choice('n_estimators', np.arange(40, 100+1, dtype=int)),
                                "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
                                "reg_lambda": hp.uniform("reg_lambda", 0.01, 10),
                                },

                   "RF":       {"max_depth": hp.choice('max_depth', list(range(10, 30+1))),
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
        for clf in self.models.values():
            clf.fit(x, y)
        self.fitted = True

    def predict(self, X_test: Union[ndarray, DataFrame]) -> ndarray:
        if not self.fitted:
            raise ValueError("isn't fitted yet")

    def cv(self, X_train: Union[ndarray, DataFrame], y_train: DataFrame) -> None:
        skf = StratifiedKFold(shuffle=True, random_state=42)

        x_cv_train, y_cv_train = None, None
        cv_res = {clf_name: {"score": 0} for clf_name in self.models}

        for clf_name in self.models:
            for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
                x_cv_train, y_cv_train = X_train.iloc[train_index], y_train.iloc[train_index].values.ravel()
                x_cv_test, y_cv_test = X_train.iloc[test_index], y_train.iloc[test_index].values.ravel()
                clf_model = deepcopy(self.models[clf_name])

                def hyperopt_objective(params):
                    # print(params)
                    _model = deepcopy(clf_model).set_params(**params)
                    _model.fit(x_cv_train, y_cv_train)
                    y_cv_pred = _model.predict(x_cv_test)
                    m_value = metrics.f1_score(
                        y_cv_test, y_cv_pred, average="macro")

                    return -m_value

                best = fmin(
                    hyperopt_objective, space=self.h_space_clf[clf_name], algo=tpe.suggest, timeout=2)
                score = -hyperopt_objective(best)
                print(cv_res)
                if score > cv_res[clf_name]["score"]:
                    cv_res[clf_name] = best
                    cv_res[clf_name]["score"] = score

        return cv_res


if __name__ == "__main__":
    c_model = ClassifierModel()
    x = DataFrame(np.random.random((100, 5)))
    y = DataFrame(np.random.randint(0, 2, 100))
    c_model.cv(x, y)
