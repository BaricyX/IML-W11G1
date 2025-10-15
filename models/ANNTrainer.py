import numpy as np

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from classifier_trainer import ClassifierTrainerQ3

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

    def train(self):
        Xtr = self.X_train.astype(np.float32).toarray()
        Xtr = self._scaler.fit_transform(Xtr)

        self.model = MLPClassifier(
        hidden_layer_sizes=(256,128,64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
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