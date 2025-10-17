from classifier_trainer import ClassifierTrainerQ3

from numpy import round, clip

from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

class LogisticRegressionTrainer(ClassifierTrainerQ3):

    def prepare_data(self):
        # if considering time feature as part of the model
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

        self.best_c = None

    # Perform cross-validation using StratifiedKFold with accuracy, selecting the optimal C value.
    def find_best_c(self, c_grid=None, cv_splits=5, random_state=2025):

        if c_grid is None:
            c_grid = [0.03, 0.1, 0.3, 1.0, 3.0]

        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        best_score, best_c = -1.0, None
        for c in c_grid:
            clf = LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                max_iter=8000,
                tol=1e-3,
                C=c,
                n_jobs=-1,
                random_state=2025
            )
            scores = cross_val_score(
                clf, self.X_train, self.y_train,
                cv=skf, scoring='accuracy'
            )
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score, best_c = mean_score, c

        self.best_c = best_c
        print(f"time_feature={self.time_feature} : best C = {self.best_c} ('Accuracy' = {best_score:.4f})")
        return self.best_c, best_score

    def train(self):
        C = self.best_c if hasattr(self, "best_c") and self.best_c is not None else 1.0
        self.model = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            max_iter=8000,
            tol=1e-3,
            n_jobs=-1,
            C=C,
            random_state=2025
        )
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_predict = self.model.predict(self.X_test)
        self.y_predict = clip(round(self.y_predict), self.y_test.min(), self.y_test.max()).astype(int)