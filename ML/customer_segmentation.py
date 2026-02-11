# Customer Segmentation using K-Means
# Data Source (example):
# https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Create Sample Dataset (150 customers)
# -----------------------------

np.random.seed(42)
n = 150

annual_spending = np.random.randint(500, 20000, size=n)
purchase_frequency = np.random.randint(1, 30, size=n)
age = np.random.randint(18, 70, size=n)

df = pd.DataFrame({
    "annual_spending": annual_spending,
    "purchase_frequency": purchase_frequency,
    "age": age
})

# -----------------------------
# Scale Features
# -----------------------------

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# -----------------------------
# Elbow Method
# -----------------------------

inertia = []

for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 8), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.savefig("elbow_plot.png")
plt.show()

# -----------------------------
# Apply K-Means (K = 3)
# -----------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled_data)

# -----------------------------
# Cluster Analysis
# -----------------------------

print("Cluster Means:")
print(df.groupby("cluster").mean())

# Save results
df.to_csv("customer_segments.csv", index=False)
