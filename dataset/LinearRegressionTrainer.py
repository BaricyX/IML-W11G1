import pandas as pd

from dataset.classifier_trainer import ClassifierTrainerQ3


class LinearRegressionTrainer(ClassifierTrainerQ3):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, time_feature: bool):
        super().__init__(train_df, test_df, time_feature)

    def prepare_data(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def outcome(self):
        pass


