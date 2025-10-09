from abc import ABC, abstractmethod

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_predict = None

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def outcome(self):
        accuracy = accuracy_score(self.y_test, self.y_predict)
        recall = recall_score(self.y_test, self.y_predict, average='macro')
        precision = precision_score(self.y_test, self.y_predict, average='macro')
        f1 = f1_score(self.y_test, self.y_predict, average='macro')

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Macro-Precision: {precision:.4f}')
        print(f'Macro-Recall: {recall:.4f}')
        print(f'Macro-F1 Score: {f1:.4f}')


class ClassifierTrainerQ3(ClassifierTrainer, ABC):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 categorical_features: list, numerical_features: list, time_feature: bool):
        super().__init__(train_df, test_df, categorical_features, numerical_features)
        self.time_feature = time_feature