from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
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

    def draw(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_predict, alpha=0.5, label='Predicted vs Actual')
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 color='red', linewidth=2, label='Ideal fit')
        plt.title('Predicted vs Actual Incident Grades')
        plt.xlabel('Actual Incident Grade')
        plt.ylabel('Predicted Incident Grade')
        plt.legend()
        plt.grid(True)
        plt.show()
    def outcome(self):
        accuracy = accuracy_score(self.y_test, self.y_predict)
        recall = recall_score(self.y_test, self.y_predict, average='macro')
        precision = precision_score(self.y_test, self.y_predict, average='macro', zero_division=0)
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

    def time_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        for df in [train_df, test_df]:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['hour'] = df['Timestamp'].dt.hour
            df['dayofweek'] = df['Timestamp'].dt.dayofweek
            df['month'] = df['Timestamp'].dt.month
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        new_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
        for c in new_cols:
            if c not in self.numerical_features:
                self.numerical_features.append(c)
        return train_df, test_df
