from typing import Dict, Union

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from numpy import ndarray
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

import hyperopt
from hyperopt import hp
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from copy import deepcopy


class Boosting:
    def __init__(self) -> None:
        pass

    def fit(**args):
        pass

    def transform(**args):
        pass

    def predict(**args):
        pass


class GBoosts:
    clf_dict = {"CatBoost": CatBoostClassifier(),
                "LGM": LGBMClassifier(),
                "XGB": XGBClassifier(),
                "RF": RandomForestClassifier()}

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
    
    h_space_clf = {"CatBoost": {"depth": hp.uniformint("depth", 4, 10),
                                "n_estimators": hp.uniformint("n_estimators", 40, 100),
                                "learning_rate": hp.uniform("learning_rate", 1e-7, 1e-2),
                                "l2_leaf_reg" : hp.uniform("l2_leaf_reg", 0.1, 10),
                                },
                   # url https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
                   "LGM": {}}

    def __init__(self) -> None:
        self.fitted = False
        self.best_models: Dict[str, object] = {}

    def fit(self, X_train: Union[ndarray, DataFrame], y_train: Union[ndarray, DataFrame]) -> None:
        pass

    def predict(self, X_test: Union[ndarray, DataFrame]) -> ndarray:
        if not self.fitted:
            raise ValueError("isn't fitted yet")

    def cv(self, X_train: Union[ndarray, DataFrame], y_train: Union[ndarray, DataFrame]) -> None:
        skf = StratifiedKFold()
        clf_ = deepcopy(self.clf_dict)
        balance_ = deepcopy(self.balance_dict)

        for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            x_cv_train, y_cv_train = X_train.iloc[train_index], y_train.iloc[train_index]


