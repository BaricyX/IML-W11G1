import pandas as pd
from matplotlib import pyplot as plt
from numpy import round, clip

from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

cat_columns = ['Category', 'EntityType', 'EvidenceRole', 'SuspicionLevel', 'LastVerdict',
               'ResourceType', 'Roles', 'AntispamDirection', 'ThreatFamily']

numerical_columns = ['DeviceId', 'Sha256', 'IpAddress', 'Url', 'AccountSid', 'AccountUpn', 'AccountObjectId',
                     'AccountName', 'DeviceName', 'NetworkMessageId', 'EmailClusterId', 'RegistryKey',
                     'RegistryValueName', 'RegistryValueData', 'ApplicationId', 'ApplicationName',
                     'OAuthApplicationId', 'FileName', 'FolderPath', 'ResourceIdName', 'OSFamily',
                     'OSVersion', 'CountryCode', 'State', 'City']

# read dataset
train_data = pd.read_csv('train.csv', low_memory=False)  # read a few rows to start
test_data = pd.read_csv('test.csv', low_memory=False)  # read a few rows to start

# pre process data
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(train_data[cat_columns])

train_data_ohe = ohe.transform(train_data[cat_columns])
test_data_ohe = ohe.transform(test_data[cat_columns])

train_data_numerical = csr_matrix(train_data[numerical_columns].fillna(-1).values)
test_data_numerical = csr_matrix(test_data[numerical_columns].fillna(-1).values)


# hstack
X_train = hstack([train_data_ohe, train_data_numerical])
X_test = hstack([test_data_ohe, test_data_numerical])

le = LabelEncoder()
le.fit(train_data['IncidentGrade'])

y_train = le.transform(train_data['IncidentGrade'])
y_test = le.transform(test_data['IncidentGrade'])

# train a model with time feature

# Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Convert continuous predictions to integer class labels
y_pred_rounded = clip(round(y_pred), y_test.min(), y_test.max()).astype(int)

accuracy = accuracy_score(y_test, y_pred_rounded)
recall = recall_score(y_test, y_pred_rounded, average='macro')
precision = precision_score(y_test, y_pred_rounded, average='macro')
f1 = f1_score(y_test, y_pred_rounded, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Macro-Precision: {precision:.4f}')
print(f'Macro-Recall: {recall:.4f}')
print(f'Macro-F1 Score: {f1:.4f}')


# # Plot the scatter plot and regression line
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
# plt.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()],
#          color='red', linewidth=2, label='Ideal fit')
# plt.title('Predicted vs Actual Incident Grades')
# plt.xlabel('Actual Incident Grade')
# plt.ylabel('Predicted Incident Grade')
# plt.legend()
# plt.grid(True)
# plt.show()

