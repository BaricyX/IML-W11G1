from abc import ABC, abstractmethod

import pandas as pd

class ClassifierTrainer(ABC):
    """
    This abstract class defines the general process techniques for training
    Microsoft Cyber Attack prediction dataset.

    - categorial features - cat_columns
    - numerical features - numerical_columns

    """
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_features: list, numerical_features: list):
        self.train_df = train_df
        self.test_df = test_df

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def outcome(self):
        pass

class ClassifierTrainerQ3(ClassifierTrainer, ABC):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, time_feature: bool):
        super().__init__(train_df, test_df)
        self.time_feature = time_feature