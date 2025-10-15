from classifier_trainer import ClassifierTrainerQ3

from numpy import round, clip

from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class LogisticRegressionTrainer(ClassifierTrainerQ3):

    def prepare_data(self):
        # if considering time feature as part of the model
        if self.time_feature:
            self.train_df, self.test_df = self.time_features(self.train_df, self.test_df)

        #convert categorical features into numerical features
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(self.train_df[self.categorical_features])


        train_data_ohe = ohe.transform(self.train_df[self.categorical_features])
        test_data_ohe = ohe.transform(self.test_df[self.categorical_features])

        train_data_num = csr_matrix(self.train_df[self.numerical_features].fillna(-1).values)
        test_data_num = csr_matrix(self.test_df[self.numerical_features].fillna(-1).values)

        self.X_train = hstack([train_data_ohe, train_data_num])
        self.X_test = hstack([test_data_ohe, test_data_num])

        le = LabelEncoder()
        le.fit(self.train_df['IncidentGrade'])
        self.y_train = le.transform(self.train_df['IncidentGrade'])
        self.y_test = le.transform(self.test_df['IncidentGrade'])

    def train(self):
        self.model = LogisticRegression(max_iter=200, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_predict = self.model.predict(self.X_test)
        self.y_predict = clip(round(self.y_predict), self.y_test.min(), self.y_test.max()).astype(int)