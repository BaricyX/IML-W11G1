from classifier_trainer import ClassifierTrainerQ3

import numpy as np

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

class ANNTrainer(ClassifierTrainerQ3):

    def prepare_data(self):
        if self.time_feature:
            self.train_df, self.test_df = self.time_features(self.train_df, self.test_df)

        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(self.train_df[self.categorical_features])

        train_data_ohe = ohe.transform(self.train_df[self.categorical_features])
        test_data_ohe  = ohe.transform(self.test_df[self.categorical_features])

        train_data_num = csr_matrix(self.train_df[self.numerical_features].fillna(-1).values)
        test_data_num  = csr_matrix(self.test_df[self.numerical_features].fillna(-1).values)

        self.X_train = hstack([train_data_ohe, train_data_num])
        self.X_test  = hstack([test_data_ohe , test_data_num])

        le = LabelEncoder()
        le.fit(self.train_df['IncidentGrade'])
        self.y_train = le.transform(self.train_df['IncidentGrade'])
        self.y_test  = le.transform(self.test_df['IncidentGrade'])

        self._scaler = StandardScaler(with_mean=True, with_std=True)
        self.best_alpha = None

    # Perform cross-validation using StratifiedKFold with accuracy, selecting the optimal alpha value.
    def find_best_alpha(self, alpha_grid=None, cv_splits=5, random_state=2025):
        if alpha_grid is None:
            alpha_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

        Xtr_dense = self.X_train.astype(np.float32).toarray()
        ytr = self.y_train

        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        best_score, best_alpha = -1.0, None
        for a in alpha_grid:
            pipe = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                MLPClassifier(
                    hidden_layer_sizes=(256,128,64),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    alpha=a,
                    batch_size=1024,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=2025
                )
            )
            scores = cross_val_score(pipe, Xtr_dense, ytr, cv=skf, scoring='accuracy')
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score, best_alpha = mean_score, a

        self.best_alpha = best_alpha
        print(f"time_feature={self.time_feature} : best alpha = {self.best_alpha:.0e} ('Accuracy' = {best_score:.4f})")
        return self.best_alpha, best_score

    def train(self):
        Xtr = self.X_train.astype(np.float32).toarray()
        Xtr = self._scaler.fit_transform(Xtr)

        alpha_to_use = self.best_alpha
        self.model = MLPClassifier(
            hidden_layer_sizes=(256,128,64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            alpha=alpha_to_use,
            batch_size=1024,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=2025
        )

        self.model.fit(Xtr, self.y_train)

    def predict(self):
        Xte = self.X_test.astype(np.float32).toarray()
        Xte = self._scaler.transform(Xte)
        self.y_predict = self.model.predict(Xte)