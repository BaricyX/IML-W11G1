import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from classifier_trainer import ClassifierTrainerQ3

from numpy import round, clip
import numpy as np


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



cat_columns = ['Category', 'EntityType', 'EvidenceRole', 'SuspicionLevel', 'LastVerdict',
               'ResourceType', 'Roles', 'AntispamDirection', 'ThreatFamily']

numerical_columns = ['DeviceId', 'Sha256', 'IpAddress', 'Url', 'AccountSid', 'AccountUpn', 'AccountObjectId',
                     'AccountName', 'DeviceName', 'NetworkMessageId', 'EmailClusterId', 'RegistryKey',
                     'RegistryValueName', 'RegistryValueData', 'ApplicationId', 'ApplicationName',
                     'OAuthApplicationId', 'FileName', 'FolderPath', 'ResourceIdName', 'OSFamily',
                     'OSVersion', 'CountryCode', 'State', 'City']

test_file = '../dataset/test.csv'
train_file = '../dataset/train.csv'

model_no_time = LogisticRegressionTrainer(
    train_df=pd.read_csv(test_file, low_memory=False),
    test_df=pd.read_csv(train_file, low_memory=False),
    categorical_features=cat_columns,
    numerical_features=numerical_columns,
    time_feature=False
)
model_no_time.prepare_data()
model_no_time.train()
model_no_time.predict()
model_no_time.outcome()


model_time = LogisticRegressionTrainer(
    train_df=pd.read_csv(train_file, low_memory=False),
    test_df=pd.read_csv(test_file, low_memory=False),
    categorical_features=cat_columns,
    numerical_features=numerical_columns,
    time_feature=True
)
model_time.prepare_data()
model_time.train()
model_time.predict()
model_time.outcome()
