import pandas as pd
from sklearn.cluster import KMeans
# import MultiLabelBinarizer, so that we can take the multi-class column Mitre Technique
# and binarize it to encode for each technique (1 = technique occurs, 0 = technique doesn't)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score, silhouette_samples
import re
from collections import Counter
from itertools import combinations

class KMeansTrainer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.col = "MitreTechniques"
        self.col2 = "IncidentGrade"
        self.X = None # this is the encoded binary vector
        self.model = None
        self.technique_classes = None # these are all the unique technique names the model has seen

    def prepare_data(self):
        # decided that missing values are out of scope, as trying to find patterns with at least two techniques
        # so first, drop missing values
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
        df = self.df.dropna(subset=[self.col]).copy() # make a new copy

        # only need mitretechniques and incident grade
        df = df[[self.col, self.col2]]
        df = df.astype(str) # just safety to ensure everything is taken as a string

        # keep only true positives
        df = df[df[self.col2] == "TruePositive"]

        # remove empty or blank MitreTechniques (there is many)
        df[self.col] = df[self.col].str.strip()  # no spaces
        df = df[df[self.col] != ""]  # no empty strings
        print(f"removed empty MitreTechniques: {len(df)} rows remaining")

        technique_lists = []
        valid_rows = []

        for i, technique_list in enumerate(df[self.col]):
            techniques = re.split(r'[;,\s]+', technique_list)
            # searching for combination patterns, so need an array of more than 1 mitre technique
            if len(techniques) <= 1:
                continue

            cleaned_techniques_list = []
            for t in techniques:
                technique = t.strip() # clean up data from whitespace etc
                if technique != "":
                    cleaned_techniques_list.append(technique)

            if len(cleaned_techniques_list) > 1:
                technique_lists.append(cleaned_techniques_list)
                valid_rows.append(i)

        df = df.iloc[valid_rows].copy() # have to make sure lengths match
        df[self.col] = technique_lists
        self.df = df
        print(f"prepared rows. Total ready: {len(df)}")
        cleaned_data = self.encode()
        return cleaned_data

    def encode(self):
        mlb = MultiLabelBinarizer()
        self.X = mlb.fit_transform(self.df[self.col]) # make binary vector for each technique
        self.technique_classes = mlb.classes_
        print(f"encoded {len(self.technique_classes)} unique mitre techniques.")
        return self.X


    def train(self, n_clusters: int):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = self.model.fit_predict(self.X)

        self.df["Cluster"] = self.labels # add back in
        print(f"train complete")

    # scikit-learn documentation for selecting best k-value:
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    def silhouette_analysis(self, k_values, save=True):
        X = self.X
        best_k = None
        best_score = -1

        for n_clusters in k_values:
            print(f"evaluating k={n_clusters}...")

            # start KMeans with n cluster value and random state
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model_labels = model.fit_predict(X)

            # the average value for all samples
            silhouette_avg = silhouette_score(X, model_labels)
            # silhouette scores
            sample_silhouette_values = silhouette_samples(X, model_labels)
            print(f"average silhouette score: {silhouette_avg:.4f}---------------------------")

            # if this k has a higher score than previous k score, store it to return
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = n_clusters

        print(f"\nBest number of clusters: {best_k} (silhouette score = {best_score:.4f})")
        return best_k

    def show_patterns(self, top_n=10):
        pattern_sizes = [2, 3, 4]
        risk_df = pd.read_csv("technique_risk_scores.csv")
        all_results = []
        # For each cluster found, print the most common MITRE techniques used together
        # pattern_size = max number of MITRE techniques that occur together frequently

        for size in pattern_sizes:
            cluster_ids = sorted(self.df["Cluster"].unique())
            for cluster_id in cluster_ids:
                # Go through one cluster incidents at a time
                subset = self.df[self.df["Cluster"] == cluster_id]
                pattern_counter = Counter()

                for technique_list in subset[self.col]:
                    # Only use incidents that have enough techniques (no patterns if technique list too low)
                    if len(technique_list) < size:
                        continue
                    # Create combinations of the chosen size (using 3)
                    patterns = combinations(sorted(technique_list), size)
                    pattern_counter.update(patterns)

                most_common_patterns = pattern_counter.most_common(top_n)

                # Now finding the associated risk rankings with each pattern
                ranked_patterns = []
                for combo, count in most_common_patterns:
                    # Get risk scores for each technique in the combo
                    scores = []
                    for t in combo:
                        # Find the technique column
                        match = risk_df[risk_df["Technique_ID"] == t]
                        if not match.empty:
                            scores.append(match["risk_score_norm"].values[0])
                    if scores:
                        avg_risk = sum(scores) / len(scores)
                    else:
                        avg_risk = 0
                    # Add the data for the row
                    all_results.append({ "pattern_size": size, "cluster": cluster_id, "techniques": " + ".join(combo),
                                         "occurrences": count, "avg_risk_score": round(avg_risk, 5) })

        results = pd.DataFrame(all_results)
        results = results.sort_values(by=["avg_risk_score", "occurrences"], ascending=[False, False])
        results.to_csv("mitre_pattern_results.csv")

if __name__ == "__main__":
    df = pd.read_csv('../dataset/train.csv')
    kmeans = KMeansTrainer(df)
    encoded_X = kmeans.prepare_data()

    # calculate the best k value
    best_k = kmeans.silhouette_analysis(k_values=range(2,9))
    kmeans.train(n_clusters=best_k)

    kmeans.show_patterns()

