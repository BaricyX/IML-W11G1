import os
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

cat_columns = ['Category', 'EntityType', 'EvidenceRole', 'SuspicionLevel', 'LastVerdict',
               'ResourceType', 'Roles', 'AntispamDirection', 'ThreatFamily']

numerical_columns = ['DeviceId', 'Sha256', 'IpAddress', 'Url', 'AccountSid', 'AccountUpn', 'AccountObjectId',
                     'AccountName', 'DeviceName', 'NetworkMessageId', 'EmailClusterId', 'RegistryKey',
                     'RegistryValueName', 'RegistryValueData', 'ApplicationId', 'ApplicationName',
                     'OAuthApplicationId', 'FileName', 'FolderPath', 'ResourceIdName', 'OSFamily',
                     'OSVersion', 'CountryCode', 'State', 'City']


def process_data():
    train_data = pd.read_csv('../dataset/train.csv', nrows=10000)  # read a few rows to start
    test_data = pd.read_csv('../dataset/test.csv', nrows=10000)  # read a few rows to start

    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(train_data[cat_columns])

    train_data_ohe = ohe.transform(train_data[cat_columns])
    test_data_ohe = ohe.transform(test_data[cat_columns])

    train_data_numerical = csr_matrix(train_data[numerical_columns].fillna(-1).values)
    test_data_numerical = csr_matrix(test_data[numerical_columns].fillna(-1).values)

    X_train = hstack([train_data_ohe, train_data_numerical])
    X_test = hstack([test_data_ohe, test_data_numerical])

    le = LabelEncoder()
    le.fit(train_data['IncidentGrade'])

    y_train = le.transform(train_data['IncidentGrade'])
    y_test = le.transform(test_data['IncidentGrade'])

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    model.fit(X_train, y_train)

    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred


# get the data
X_train, y_train, X_test, y_test = process_data()

# train a model
model = train_model(X_train, y_train)

# make predictions
y_pred = predict(model, X_test)

# evaluate performance
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print('Accuracy: {}'.format(accuracy))
print('Macro-Precision: {}'.format(precision))
print('Macro-Recall: {}'.format(recall))
print('Macro-F1 Score: {}'.format(f1))
