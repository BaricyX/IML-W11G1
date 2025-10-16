import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

class DecisionTreelassifier:
    def __init__(self, feature_groups, target="IncidentGrade", max_depth=10, min_samples_split=10):
        self.feature_groups = feature_groups
        self.target = target
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.le = LabelEncoder()
        self.model = DecisionTreeClassifier(
            random_state=42, 
            max_depth=self.max_depth, 
            min_samples_split=self.min_samples_split
        )

    def load_data(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path, low_memory=False)
        self.test_data = pd.read_csv(test_path, low_memory=False)
        # Select the features that we want in the dataset
        self.selected_features = [
            f for features in self.feature_groups.values() 
            for f in features if f in self.train_data.columns
        ]

    def prepare_data(self):
        # First separate categorical and numerical features
        cat_cols = [c for c in self.selected_features if self.train_data[c].dtype == "object" or self.train_data[c].dtype == "bool"]
        num_cols = [c for c in self.selected_features if c not in cat_cols]

        # Use OneHotEncoder for the categorical features in the training dataset
        self.ohe.fit(self.train_data[cat_cols])
        X_train_cat = self.ohe.transform(self.train_data[cat_cols])
        X_test_cat = self.ohe.transform(self.test_data[cat_cols])

        # Clean data
        X_train_num = csr_matrix(self.train_data[num_cols].fillna(-1).values)
        X_test_num = csr_matrix(self.test_data[num_cols].fillna(-1).values)

        # Use hstack to combine the categorical and numerical features
        self.X_train = hstack([X_train_cat, X_train_num])
        self.X_test = hstack([X_test_cat, X_test_num])

        # Encode the target variable
        self.y_train = self.le.fit_transform(self.train_data[self.target])
        self.y_test = self.le.transform(self.test_data[self.target])

        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(self.y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="macro", zero_division=0)

        print("=== Decision Tree Performance ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.le.classes_))

    def feature_importance(self):
        # First retrieve the feature's names after the OneHotEncoding
        encoded_features = self.ohe.get_feature_names_out(self.cat_cols)
        all_features = list(encoded_features) + self.num_cols

        # Compute feature importance
        importances = pd.DataFrame({
            "feature": all_features,
            "importance": self.model.feature_importances_
        })
        importances = importances[importances["importance"] > 0]

        # Map the features to their corresponding categories
        feature_to_group = {f: g for g, feats in self.feature_groups.items() for f in feats}

        def map_group(f):
            for col, grp in feature_to_group.items():
                if col in f:
                    return grp
            return None

        importances["group"] = importances["feature"].apply(map_group)

        # Aggregate each category's importance
        group_importances = importances.groupby("group")["importance"].sum().sort_values(ascending=False).reset_index()
        group_importances["percentage"] = 100 * group_importances["importance"] / group_importances["importance"].sum()
        print("\n=== Group-Level Feature Importances ===")
        print(group_importances)

        # Plot horizontal bar plot
        plt.figure(figsize=(8, 5))
        plt.barh(group_importances["group"], group_importances["percentage"], color="steelblue")
        plt.gca().invert_yaxis()
        plt.title("Decision Tree Feature Importance by Feature Group")
        plt.xlabel("Importance Percentage (%)")
        plt.ylabel("Feature Group")
        plt.tight_layout()
        plt.show()

# Main entry point
feature_groups = {
    "Organization": ["OrgId"],
    "Detection": ["DetectorId", "AlertTitle", "Category"],
    "Entity": ["EntityType", "EvidenceRole", "Roles", "OSFamily", "OSVersion"],
    "Account": ["AccountSid", "AccountUpn", "AccountObjectId", "AccountName"],
    "Network_Resource": ["IpAddress", "Url", "ThreatFamily"],
    "Location": ["CountryCode", "State", "City"]
}

classifier = DecisionTreelassifier(feature_groups)
classifier.load_data("./dataset/train.csv", "./dataset/test.csv")
classifier.prepare_data()
classifier.train()
classifier.predict()
classifier.feature_importance()
