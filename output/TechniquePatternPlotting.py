# This is the script for generating the plots for question 2's results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# I have hard-coded in the resulting csv from KMeans.py
df = pd.read_csv("mitre_pattern_results copy.csv")

sns.set_theme(style="whitegrid", font_scale=1.1)

# Plot top 10 risky combinations overall
top10 = df.sort_values(by="avg_risk_score", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top10, x="avg_risk_score", y="techniques", palette="Reds_r")
plt.title("Top 10 Riskiest Technique Combinations")
plt.xlabel("Average Risk Score")
plt.ylabel("Technique Combination")
plt.tight_layout()
plt.savefig("top10_riskiest_patterns.png", dpi=300)
plt.show()

# Plot average risk by cluster
plt.figure(figsize=(8, 6))
cluster_avg = df.groupby("cluster")["avg_risk_score"].mean().reset_index()
sns.barplot(data=cluster_avg, x="cluster", y="avg_risk_score", palette="Purples_d")
plt.title("Average Risk by Cluster")
plt.xlabel("Cluster ID")
plt.ylabel("Average Risk Score")
plt.tight_layout()
plt.savefig("risk_by_cluster.png", dpi=300)
plt.show()

# Plot top patterns per cluster
top_per_cluster = df.sort_values(['cluster', 'avg_risk_score'], ascending=[True, False]).groupby('cluster').head(5)
plt.figure(figsize=(10, 7))
sns.barplot(data=top_per_cluster, x="avg_risk_score", y="techniques", hue="cluster", dodge=False)
plt.title("Top 5 Risky Patterns per Cluster")
plt.xlabel("Average Risk Score")
plt.ylabel("Technique Combination")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("top_risky_patterns_per_cluster.png", dpi=300)
plt.show()
