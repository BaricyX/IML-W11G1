import pandas as pd
from kmodes.kmodes import KModes
from sklearn.preprocessing import MultiLabelBinarizer
import re
from collections import Counter
from itertools import combinations

class KModesTrainer:

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
        self.model = KModes(n_clusters=n_clusters, random_state=42)
        self.labels = self.model.fit_predict(self.X)

        self.df["Cluster"] = self.labels # add back in
        print(f"train complete")

    def find_best(self, k_values=range(2,9)):
        sum_of_distances = []
        for k in k_values:
            kmodes = KModes(n_clusters=k, n_init=5, random_state=42, verbose=0)
            kmodes.fit(self.X)
            distance = kmodes.cost_
            sum_of_distances.append(distance)
            print(f"k = {k}, cost = {distance:.2f}")

        # now have to figure out how much the distance sum drops when k is increased
        decreases = []
        for i in range(1, len(sum_of_distances)):
            decrease = sum_of_distances[i - 1] - sum_of_distances[i]  # previous cost - current cost
            decreases.append(decrease)

        print("Distance drops between each k:", decreases)

        # find where it stops being beneficial to increase k
        smallest_drop_index = decreases.index(min(decreases))
        best_k = list(k_values)[smallest_drop_index + 1]

        print(f"Best k: {best_k}")
        return best_k

    def show_patterns(self, top_n=10, pattern_size=2):
        # for each cluster found, print the most common MITRE techniques used together
        # pattern size = max number of MITRE techniques that occur together frequently
        cluster_ids = sorted(self.df["Cluster"].unique())

        for cluster_id in cluster_ids:
            print(f"Cluster {cluster_id}:")

            # go through one cluster incidents at ta time
            subset = self.df[self.df["Cluster"] == cluster_id]

            pattern_counter = Counter()

            for technique_list in subset[self.col]:
                # only use incidents that have enough techniques (no patterns if technique list too low)
                if len(technique_list) < pattern_size:
                    continue

                # create combinations of the chosen size (using 3)
                patterns = combinations(sorted(technique_list), pattern_size)
                pattern_counter.update(patterns)

            most_common_patterns = pattern_counter.most_common(top_n)

            print(f"Top {top_n} technique combinations (size={pattern_size}):")
            for combo, count in most_common_patterns:
                technique_combination = " + ".join(combo)
                print(f"  {technique_combination}: {count} times")

            print(f"Total unique combinations of size {pattern_size}: {len(pattern_counter)}")


if __name__ == "__main__":
    df = pd.read_csv('dataset/train.csv')
    kmodesM = KModesTrainer(df)
    encoded_X = kmodesM.prepare_data()

    # calculate the best k value
    best_k = kmodesM.find_best(k_values=range(2,9))
    kmodesM.train(n_clusters=best_k)
    kmodesM.show_patterns(top_n=10)
