from abc import ABC

import pandas as pd


class classifier_trainer(ABC):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.train_df = train_df
        self.test_df = test_df

    def train(self):
        ...
    def test(self):
        ...
    def outcome(self):
        ...
