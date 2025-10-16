# This file handles the feature engineering, to be later used with the KMeans model in models/.
# Citation: used all_techniques.csv from https://github.com/0xmahmoudJo0/MITRE-Analysis-Prioritization/tree/main a study
# cited in the report

import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler

# HELPER FUNCTIONS
# This makes the column into a stringified list of dicts, so can extract the technique id safely (ex: T1110)
# Citation: I had to use ChatGPT to help me understand how to use ast and correctly identify columns. It was used for this function, then I was able to write the remaining functions on my own.
def extract_attack_id(refs):
    try:
        items = ast.literal_eval(refs)
        if isinstance(items, list):
            for ref in items:
                if "external_id" in ref and ref["source_name"] == "mitre-attack":
                    return ref["external_id"]
    except:
        pass
    return None

# This counts the number of data sources that can detect a technique
def count_data_sources(val):
    try:
        items = ast.literal_eval(val)
        if isinstance(items, list):
            return len(items)
    except:
        pass
    return 0

# From https://www.cyber.gov.au/business-government/asds-cyber-security-frameworks/ism/cybersecurity-terminology glossary,
# Asked ChatGPT to extract the possible words related to detection
detection_keywords = ["intrusion detection", "intrusion prevention", "monitor", "telemetry", "log", "logging", "event",
                      "alert", "audit", "firewall", "network traffic", "continuous monitoring", "gateway", "ids", "ips",
                      "security posture", "detection", "anomaly", "flow", "signature", "incident", "analysis",
                      "correlation"]

# This function determines how many detection keywords are present from a provided text
def keyword_strength(text):
    if not isinstance(text, str):
        return 0
    text = text.lower()
    count = sum(text.count(word) for word in detection_keywords)
    return count

impact_types = ["Confidentiality", "Integrity", "Availability"]

# This function counts the appearances of the CIA triad in a column
def count_cia_impacts(val):
    try:
        if pd.isna(val):
            return 0
        items = ast.literal_eval(val)
        if isinstance(items, list):
            # remove empty or invalid items
            items = [x for x in items if x in ["Confidentiality", "Integrity", "Availability"]]
            return len(items)
    except:
        pass
    return 0

permission_weights = {"User": 1, "Administrator": 2, "SYSTEM": 4, "root": 8}

# This function weights all the different permissions according to the hierarchy
def weight_permissions(val, permission_weights):
    try:
        items = ast.literal_eval(val)
        if isinstance(items, list) and len(items) > 0:
            weights = [permission_weights.get(x, 0) for x in items]
            return sum(weights) / len(weights)
    except:
        pass
    return 0

# This function counts the defenses bypassed in a column
def count_defenses_bypassed(val):
    if pd.isna(val):
        return 0
    try:
        items = ast.literal_eval(val)
        if isinstance(items, list):
            return len(items)
    except:
        pass
    return 0

# BEGIN ----------------------------------------------------------------------------------------------------------------
df = pd.read_csv("all_techniques.csv")

# Cleaning the data ----------------------------------------------------------------------------------------------------
# Filter out irrelevant mitre domains
df = df[df["x_mitre_domains"].str.contains("enterprise-attack", na=False)]
# Focus only on Windows platform, as we are using a Microsoft dataset
df = df[df["x_mitre_platforms"].str.contains("Windows", na=False)]
# Extract the Technique ID from external_references column
df["Technique_ID"] = df["external_references"].apply(extract_attack_id)

# Keep only the columns needed for determining detection, severity, mitigation potential and difficulty scores
keep_columns = ["Technique_ID", "name", "description", "x_mitre_impact_type", "x_mitre_detection",
                "x_mitre_defense_bypassed", "x_mitre_permissions_required", "x_mitre_data_sources"]
df = df[keep_columns]
# As a fallback, drop rows that are missing Technique IDs
df = df.dropna(subset="Technique_ID")

# DETECTION -------------------------------------------------------------------------------------------
# data source count + detection keywords count
# First, determine how many data sources can detect the technique, and add it to the data frame
df["data_source_count"] = df["x_mitre_data_sources"].apply(count_data_sources)
# Now determine how many detection keywords were mentioned, and add it to the data frame
df["detection_keyword_count"] = df["x_mitre_detection"].apply(keyword_strength)

# Now normalize these numeric features to be 0-1 range, as mentioned in class
# Chose minmax because it does just this
scaler = MinMaxScaler()
df[["data_source_norm", "keyword_norm"]] = scaler.fit_transform(df[["data_source_count", "detection_keyword_count"]])
# I chose to weight the data sources more heavily because it is standardized, and the keywords are descriptive but varies
# across techniques
df["detection_score"] = 0.7 * df["data_source_norm"] + 0.3 * df["keyword_norm"]

df["undetectability"] = 1 - df["detection_score"]

# SEVERITY-------------------------------------------------------------
# First, determine penalty for impacts encountered
# For impact type, I penalize based on the present of one of the CIA triad - as for different organizations, or types of
# products in Microsoft's case, confidentiality, integrity, and availability may be weighted differently and it would be
# biased to weight these numerically.
# https://www.fortinet.com/resources/cyberglossary/cia-triad
df["impact_count"] = df["x_mitre_impact_type"].apply(count_cia_impacts)

# Now determine penalty score (divide by 3 as there are 3 maximum penalties, this also scales it)
df["impact_penalty"] = df["impact_count"] / 3

# For permissions, I follow the Analytic Hierarchy Process (AHP) from https://arxiv.org/pdf/1812.11404 (cited in report)
# user -> Administrator -> system -> root
# minimal privileges -> broad system access -> full os control, kernel level -> total dominance
# just using exponential progression 1, 2, 4, 8
# and then combine hierarchy severity weights with frequency
df["permission_score_unnormalized"] = df["x_mitre_permissions_required"].apply(lambda x: weight_permissions(x, permission_weights))
df[["permission_score"]] = scaler.fit_transform(df[["permission_score_unnormalized"]])

# Both high impact and high privilege make a technique severe
df["severity"] = df["impact_penalty"] * df["permission_score"]

# MITIGATION POTENTIAL-----------------------------------------------------
# First count how many defenses can be bypassed
df["defenses_bypassed_count"] = df["x_mitre_defense_bypassed"].apply(count_defenses_bypassed)

# If there has been more defenses bypassed, then it has lower mitigation.
max_bypassed = df["defenses_bypassed_count"].max()
# This is the data normalization formula simplified: x - min(x) / max(x) - min(x)
df["mitigation"] = 1 - (df["defenses_bypassed_count"] / max_bypassed)
# Normalize for consistency
df[["mitigation_potential"]] = scaler.fit_transform(df[["mitigation"]])

df["unmitigatability"] = 1 - df["mitigation_potential"]

# HEURISTIC --------------------------------------------------------------------
df["risk_score"] = (0.6 * df["severity"] + 0.4 * df["undetectability"] + 0.2 * df["unmitigatability"])

# Normalize to be between 0-1
df[["risk_score_norm"]] = scaler.fit_transform(df[["risk_score"]])
df.to_csv("technique_risk_scores.csv", index=False)
