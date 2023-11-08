from typing import Any, Dict, Literal, Union

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from numpy import ndarray
from numpy.random import RandomState
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.tree import DecisionTreeClassifier

import hyperopt
from hyperopt import hp
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from copy import deepcopy


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
    clf_dict = {"CatBoost": CatBoostClassifier(),
                "LGM": LGBMClassifier(),
                "XGB": XGBClassifier(),
                "RF": RandomForestClassifier(random_state=0),
                "AdaBoost": AdaBoostClf(random_state=0)}

    h_space_clf = {"CatBoost": {"depth": hp.uniformint("depth", 4, 10),
                                "n_estimators": hp.uniformint("n_estimators", 40, 100),
                                "learning_rate": hp.uniform("learning_rate", 1e-5, 1e-1),
                                "l2_leaf_reg": hp.uniform("l2_leaf_reg", 0.01, 10),
                                },

                   "LGM":      {"max_depth": hp.uniformint("max_depth", 4, 10),
                                "n_estimators": hp.uniformint("n_estimators", 40, 100),
                                "learning_rate": hp.uniform("learning_rate", 1e-5, 1e-1),
                                "reg_lambda": hp.uniform("reg_lambda", 0.01, 10),
                                },

                   "XGB":      {"max_depth": hp.uniformint("max_depth", 4, 10),
                                "n_estimators": hp.uniformint("n_estimators", 40, 100),
                                "learning_rate": hp.uniform("learning_rate", 1e-5, 1e-1),
                                "reg_lambda": hp.uniform("reg_lambda", 0.01, 10),
                                },

                   "RF":       {"max_depth": hp.uniformint("max_depth", 4, 10),
                                "n_estimators": hp.uniformint("n_estimators", 40, 100),
                                },

                   "AdaBoost": {"max_depth": hp.uniformint("max_depth", 4, 10),
                                "n_estimators": hp.uniformint("n_estimators", 40, 100),
                                "learning_rate": hp.uniform("learning_rate", 1e-5, 1e-1)
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

    def cv(self, X_train: Union[ndarray, DataFrame], y_train: Union[ndarray, DataFrame]) -> None:
        skf = StratifiedKFold()
        
        x_cv_train, y_cv_train = None, None

        for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            x_cv_train, y_cv_train = X_train.iloc[train_index], y_train.iloc[train_index]
