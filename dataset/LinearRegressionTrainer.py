import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from classifier_trainer import ClassifierTrainerQ3

import pandas as pd
from numpy import round, clip

from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class LinearRegressionTrainer(ClassifierTrainerQ3):
    def prepare_data(self):
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(self.train_df[self.categorical_features])
        train_data_ohe = ohe.transform(self.train_df[self.categorical_features])
        test_data_ohe = ohe.transform(self.test_df[self.categorical_features])
        train_data_numerical = csr_matrix(self.train_df[self.numerical_features].fillna(-1).values)
        test_data_numerical = csr_matrix(self.test_df[self.numerical_features].fillna(-1).values)
        self.X_train = hstack([train_data_ohe, train_data_numerical])
        self.X_test = hstack([test_data_ohe, test_data_numerical])

        le = LabelEncoder()
        le.fit(self.train_df['IncidentGrade'])
        self.y_train = le.transform(self.train_df['IncidentGrade'])
        self.y_test = le.transform(self.test_df['IncidentGrade'])

    def train(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_predict = self.model.predict(self.X_test)
        self.y_predict = clip(round(self.y_predict), self.y_test.min(), self.y_test.max()).astype(int)

    def outcome(self):
        pass
cat_columns = ['Category', 'EntityType', 'EvidenceRole', 'SuspicionLevel', 'LastVerdict',
               'ResourceType', 'Roles', 'AntispamDirection', 'ThreatFamily']

numerical_columns = ['DeviceId', 'Sha256', 'IpAddress', 'Url', 'AccountSid', 'AccountUpn', 'AccountObjectId',
                     'AccountName', 'DeviceName', 'NetworkMessageId', 'EmailClusterId', 'RegistryKey',
                     'RegistryValueName', 'RegistryValueData', 'ApplicationId', 'ApplicationName',
                     'OAuthApplicationId', 'FileName', 'FolderPath', 'ResourceIdName', 'OSFamily',
                     'OSVersion', 'CountryCode', 'State', 'City']

model = LinearRegressionTrainer(
    train_df=pd.read_csv('train.csv', low_memory=False),
    test_df=pd.read_csv('test.csv', low_memory=False),
    categorical_features=cat_columns,
    numerical_features=numerical_columns,
    time_feature=True
)
model.train()
model.predict()
model.outcome()
