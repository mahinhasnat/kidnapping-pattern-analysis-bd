"""
AI/ML Analysis Pipeline — Child Kidnapping Case Study (Bangladesh)
====================================================================
Input : case_study_dataset.csv (case-level, see generate_dataset.py)
Output: printed EDA summary + charts (PNG) + a simple risk-clustering model

Steps:
  1. Load & clean data
  2. Descriptive analysis (age range, division, month, relation-to-accused)
  3. K-Means clustering to find "case profile" groups
  4. Simple classifier: predict outcome risk from case features
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("case_study_dataset.csv")

# ---------- 1. Age-range bucketing ----------
bins = [0, 5, 9, 13, 17]
labels = ["0-5", "6-9", "10-13", "14-17"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

print("=" * 60)
print("MOST AFFECTED AGE RANGE")
print("=" * 60)
age_counts = df["age_group"].value_counts().sort_index()
print(age_counts)
print(f"\n>> Highest-risk age group: {age_counts.idxmax()} ({age_counts.max()} cases, "
      f"{age_counts.max()/len(df)*100:.1f}% of total)")

# ---------- 2. Descriptive breakdowns ----------
print("\n" + "=" * 60)
print("DIVISION-WISE CASE COUNT")
print("=" * 60)
print(df["division"].value_counts())

print("\n" + "=" * 60)
print("RELATION TO ACCUSED")
print("=" * 60)
print(df["relation_to_accused"].value_counts())
known_pct = (df["relation_to_accused"] != "Stranger").mean() * 100
print(f"\n>> {known_pct:.1f}% of cases involve someone known to the victim")

print("\n" + "=" * 60)
print("MONTHLY TREND (seasonality)")
print("=" * 60)
print(df["month"].value_counts().sort_index())

# ---------- 3. Charts ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

age_counts.plot(kind="bar", ax=axes[0, 0], color="#c0392b")
axes[0, 0].set_title("Cases by Age Group")
axes[0, 0].set_ylabel("Number of cases")

df["division"].value_counts().plot(kind="bar", ax=axes[0, 1], color="#2980b9")
axes[0, 1].set_title("Cases by Division")

df["month"].value_counts().sort_index().plot(kind="line", marker="o", ax=axes[1, 0], color="#27ae60")
axes[1, 0].set_title("Monthly Trend")
axes[1, 0].set_xlabel("Month")

df["relation_to_accused"].value_counts().plot(kind="barh", ax=axes[1, 1], color="#8e44ad")
axes[1, 1].set_title("Relation to Accused")

plt.tight_layout()
plt.savefig("eda_charts.png", dpi=150)
print("\nSaved eda_charts.png")

# ---------- 4. K-Means clustering: case profile groups ----------
features = df[["age", "month"]].copy()
cat_cols = ["division", "relation_to_accused", "mode_of_abduction", "gender"]
le_map = {}
for c in cat_cols:
    le = LabelEncoder()
    features[c] = le.fit_transform(df[c])
    le_map[c] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\n" + "=" * 60)
print("CASE PROFILE CLUSTERS (KMeans, k=4)")
print("=" * 60)
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    print(f"\nCluster {c} — {len(sub)} cases")
    print(f"  Avg age: {sub['age'].mean():.1f}")
    print(f"  Top division: {sub['division'].mode()[0]}")
    print(f"  Top relation: {sub['relation_to_accused'].mode()[0]}")
    print(f"  Top mode: {sub['mode_of_abduction'].mode()[0]}")
    print(f"  Most common outcome: {sub['outcome'].mode()[0]}")

# ---------- 5. Simple risk classifier ----------
# Predict whether a case outcome is high-risk ("Still missing" or "Deceased") vs "Rescued/Recovered"
df["high_risk"] = df["outcome"].isin(["Still missing", "Deceased"]).astype(int)

X = features.copy()
y = df["high_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n" + "=" * 60)
print("RISK CLASSIFIER PERFORMANCE (RandomForest)")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["Recovered", "High risk"]))

importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importance for predicting high-risk outcome:")
print(importances)

df.to_csv("case_study_dataset_analyzed.csv", index=False)
print("\nSaved case_study_dataset_analyzed.csv (with age_group, cluster, high_risk columns)")
