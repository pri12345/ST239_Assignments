# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# path + data
df = pd.read_csv('Q2_cities.csv')
cities = df['city'].values
X_raw = df.drop(columns=['city'])


# ==============================================================
# EDA
# ==============================================================

# Histograms of all features
X_raw.hist(bins=25, figsize=(18, 12), color='steelblue', edgecolor='white')
plt.suptitle('Distribution of City Features', fontsize=16)
plt.tight_layout()
plt.show()

# The population, area and density variables are heavily right-skewed,
# which will be relevant for the preprocessing step below.

# Correlation matrix
plt.figure(figsize=(14, 11))
corr = X_raw.corr()
sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm', center=0,
            linewidths=0.4, annot_kws={'size': 7})
plt.title('Correlation Matrix — City Features', fontsize=14)
plt.tight_layout()
plt.show()

# Some clear patterns:
# gdp_per_capita and avg_salary are tightly correlated (~0.9)
# tech_employment and innovation_index move together
# manufacturing_employment is negatively correlated with most economic indicators


# ==============================================================
# Preprocessing
# ==============================================================

# Log-transform right-skewed variables before scaling.
# population, area_km2, density_per_km2 and startup_density all span
# multiple orders of magnitude — log squishes those long tails in.
skewed = ['population', 'area_km2', 'density_per_km2', 'startup_density_per_100k']
X_proc = X_raw.copy()
X_proc[skewed] = np.log1p(X_raw[skewed])   # log1p just in case any 0s sneak in

# Standardise everything — PCA and distance-based methods are sensitive to scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_proc)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_raw.columns)


# ==============================================================
# PCA
# ==============================================================

pca_full = PCA()
pca_full.fit(X_scaled)

# Scree plot — how much variance does each PC explain?
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(range(1, len(explained)+1), explained * 100, color='steelblue', alpha=0.8, label='Individual')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance (%)', color='steelblue')

ax2 = ax1.twinx()
ax2.plot(range(1, len(cumulative)+1), cumulative * 100, color='crimson',
         marker='o', markersize=4, label='Cumulative')
ax2.set_ylabel('Cumulative Variance (%)', color='crimson')
ax2.axhline(80, color='crimson', linestyle='--', alpha=0.4, linewidth=1)

plt.title('PCA Scree Plot')
fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
plt.tight_layout()
plt.show()

# First 5 PCs capture the bulk of variance — we'll use 2 for visualisation
# and 5 for clustering (to keep more structure)
print("Variance explained by first 5 PCs:", np.round(cumulative[:5] * 100, 1))

# Fit the 2-component version for plotting
pca2 = PCA(n_components=2)
coords2 = pca2.fit_transform(X_scaled)

# Fit the 5-component version for clustering
pca5 = PCA(n_components=5)
coords5 = pca5.fit_transform(X_scaled)

# PC loadings — what does each component actually represent?
loadings = pd.DataFrame(
    pca2.components_.T,
    index=X_raw.columns,
    columns=['PC1', 'PC2']
).round(3)

print("\nTop PC1 loadings (sorted):")
print(loadings['PC1'].sort_values(key=abs, ascending=False).head(8))
print("\nTop PC2 loadings (sorted):")
print(loadings['PC2'].sort_values(key=abs, ascending=False).head(8))

# Loadings heatmap for PC1 and PC2
plt.figure(figsize=(8, 7))
sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
            linewidths=0.5)
plt.title('PCA Loadings — PC1 & PC2')
plt.tight_layout()
plt.show()


# ==============================================================
# Hierarchical Clustering
# ==============================================================

# Ward linkage minimises within-cluster variance at each merge step —
# it tends to give compact, similarly-sized clusters which suits this dataset well
Z = linkage(coords5, method='ward')

plt.figure(figsize=(14, 5))
dendrogram(Z, no_labels=True, color_threshold=9)
plt.axhline(y=9, color='crimson', linestyle='--', linewidth=1, label='cut at 9')
plt.title('Hierarchical Clustering Dendrogram (Ward, PCA space)')
plt.xlabel('Cities')
plt.ylabel('Ward Distance')
plt.legend()
plt.tight_layout()
plt.show()

# The dendrogram suggests 4 main branches — we'll go with k=4

hier_labels = AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(coords5)


# ==============================================================
# K-Means Clustering
# ==============================================================

# Elbow + silhouette to pick k
inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    km.fit(coords5)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(coords5, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(k_range, inertias, marker='o', color='steelblue')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(k_range, sil_scores, marker='o', color='darkorange')
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Scores')

plt.suptitle('K-Means: Choosing k', fontsize=13)
plt.tight_layout()
plt.show()

print("Silhouette scores:", {k: round(s, 3) for k, s in zip(k_range, sil_scores)})

# k=4 sits at the elbow and gives a solid silhouette — matches the dendrogram
km_final = KMeans(n_clusters=4, n_init=30, random_state=42)
km_labels = km_final.fit_predict(coords5)


# ==============================================================
# DBSCAN
# ==============================================================

# DBSCAN doesn't need a pre-specified k but it is sensitive to eps.
# We can use a k-distance plot to guide eps selection.
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5).fit(coords5)
distances, _ = nbrs.kneighbors(coords5)
k_dist = np.sort(distances[:, -1])   # 5th nearest neighbour distance, sorted

plt.figure(figsize=(8, 4))
plt.plot(k_dist, color='steelblue')
plt.axhline(y=1.8, color='crimson', linestyle='--', label='eps = 1.8')
plt.xlabel('Cities (sorted)')
plt.ylabel('Distance to 5th nearest neighbour')
plt.title('K-Distance Plot (k=5)')
plt.legend()
plt.tight_layout()
plt.show()

# Run DBSCAN with eps=1.8, min_samples=5
db = DBSCAN(eps=1.8, min_samples=5)
db_labels = db.fit_predict(coords5)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
print(f"\nDBSCAN found {n_clusters_db} clusters, {n_noise} noise points")
print("Cluster sizes:", pd.Series(db_labels).value_counts().sort_index().to_dict())


# ==============================================================
# Visualisation — all three methods side by side in PCA space
# ==============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

method_labels = [hier_labels, km_labels, db_labels]
titles = ['Hierarchical (k=4)', 'K-Means (k=4)', 'DBSCAN']
palettes = ['tab10', 'tab10', 'tab10']

for ax, labels, title in zip(axes, method_labels, titles):
    unique = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(unique)))
    for cl, col in zip(unique, colors):
        mask = labels == cl
        label_str = f'Cluster {cl}' if cl != -1 else 'Noise'
        ax.scatter(coords2[mask, 0], coords2[mask, 1], c=[col],
                   label=label_str, s=40, alpha=0.8, edgecolors='white', linewidths=0.3)
    ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=1.2)

plt.suptitle('Clustering Results in PCA Space (PC1 vs PC2)', fontsize=13)
plt.tight_layout()
plt.show()


# ==============================================================
# Cluster Profiling (using K-Means labels as the main result)
# ==============================================================

df_clustered = X_raw.copy()
df_clustered['cluster'] = km_labels
df_clustered['city'] = cities

# Mean of each feature per cluster
profile = df_clustered.groupby('cluster')[X_raw.columns].mean().round(2)
print("\nCluster profiles (feature means):\n", profile.T.to_string())

# Standardised profile heatmap — z-score against the whole dataset so
# each cluster's mean is expressed relative to the overall average
profile_z = (profile - X_raw.mean()) / X_raw.std()

plt.figure(figsize=(13, 8))
sns.heatmap(profile_z.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.4, annot_kws={'size': 8})
plt.title('Cluster Profiles — Standardised Feature Means (K-Means, k=4)')
plt.xlabel('Cluster')
plt.tight_layout()
plt.show()

# Cluster sizes
print("\nCluster sizes:")
print(pd.Series(km_labels).value_counts().sort_index())

# Sample cities per cluster
for cl in sorted(df_clustered['cluster'].unique()):
    sample = df_clustered[df_clustered['cluster'] == cl]['city'].values[:6]
    print(f"\nCluster {cl} sample cities: {', '.join(sample)}")
