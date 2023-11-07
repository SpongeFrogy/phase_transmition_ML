from typing import Literal
from sklearn.preprocessing import MinMaxScaler, Normalizer
import pandas as pd
import numpy as np

from copy import deepcopy



class NoNormalizerError(Exception):
    "only minmax and norm"


class PreprocessingError(Exception):
    "if transformed DataFrame has nan values"


class PreprocessingModel:

    @classmethod
    def do_drop(cls, col: pd.Series, p: float = 0.05) -> bool:
        """return True if col has more than p proportion of nan values and False if not

        Args:
            col (pd.Series): column of DataFrame
            p (float, optional): cut off proportion. Defaults to 0.05.

        Returns:
            bool: 
        """
        tf, counts = np.unique(col.isna(), return_counts=True)
        if len(counts) == 1:
            if tf[0] == True:
                return True
            else:
                return False
        f_pos = 0 if tf[0] == False else 1
        if counts[(f_pos + 1) % 2]/counts[f_pos] > p:
            return True
        else:
            return False

    @classmethod
    def check_transforms(cls, transformed_x: pd.DataFrame) -> None:
        if not np.all((transformed_x == transformed_x.dropna()).values):
            raise PreprocessingError("wrong fillna")

    def __init__(self, p_drop: float = 0.05, normalizer: Literal["minmax", "normalizer"] = "normalizer") -> None:
        self.p_drop = p_drop

        if normalizer not in ("minmax", "normalizer"):
            raise NoNormalizerError("only minmax and norm")

        self.normalizer = Normalizer() if normalizer == "normalizer" else MinMaxScaler()

    def fit_transform(self, x_train: pd.DataFrame) -> pd.DataFrame:
        """fit preprocessing model and transform train DataFrame with drop and filling missing and normalizer

        Args:
            x_train (pd.DataFrame): x for fit and transform.
        """

        self.cols = [name for name in x_train.columns if not self.do_drop(
            x_train[name], self.p_drop)]
        self.index = x_train.index
        
        transformed = x_train[self.cols]

        self.means = transformed.mean()
        transformed = transformed.fillna(self.means)
        self.check_transforms(transformed)

        transformed = self.normalizer.fit_transform(transformed)

        return pd.DataFrame(transformed, columns=self.cols, index = self.index)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """transform x and y 

        Args:
            x (pd.DataFrame): x to transform 

        Returns:
            pd.DataFrame: transformed x.
        """

        transformed_x = deepcopy(x[self.cols]).fillna(self.means)

        self.check_transforms(transformed_x)

        return pd.DataFrame(self.normalizer.transform(transformed_x), columns=self.cols, index=x.index)

if __name__ == "__main__":
    df = pd.DataFrame([[1, 2], [1, np.nan]])
    model = PreprocessingModel()
    model.fit_transform(df)