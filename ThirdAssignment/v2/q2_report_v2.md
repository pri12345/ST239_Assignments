# Question 2: Exploring Urban Profiles Using Dimensionality Reduction and Clustering

---

## Introduction

The dataset `Q2_cities.csv` contains 180 fictitious cities described across 20 numerical variables covering five broad themes: demographics (population, area, density, median age), economic performance (GDP per capita, average salary, unemployment rate, degree attainment), infrastructure (transit score, housing cost, commute time), environmental characteristics (green space, air quality, renewable energy), and sectoral composition (tech, manufacturing, tourism, innovation, startup density).

The aim is to find out whether these cities group naturally into coherent urban profiles. To do this, I first reduce the dimensionality of the data using Principal Component Analysis (PCA), and then apply three clustering methods — hierarchical clustering, k-means, and DBSCAN — to check whether a consistent grouping structure emerges across approaches.

---

## Exploratory Data Analysis

### Summary statistics and distributions

The dataset is complete with no missing values, and all variables are numerical. Summary statistics already hint at a lot of variation: population ranges from around 80,000 to 14 million, area from 35 to nearly 1,000 km², and GDP per capita spans a wide range too. Index variables like transit score and innovation are more evenly spread, with standard deviations roughly 15-25% of their means.

[FIGURE PLACEHOLDER: histograms of all 20 features — first plot block in q2workspace_v2.py]

The histograms confirm that `population`, `area_km2`, `density_per_km2`, and `startup_density_per_100k` are strongly right-skewed, with most cities sitting at lower values and a long tail of much larger ones. Everything else is roughly bell-shaped, which is useful to know before preprocessing.

### Pairwise relationships

[FIGURE PLACEHOLDER: correlation heatmap — second plot block in q2workspace_v2.py]

The correlation matrix reveals a few clear patterns. GDP per capita and average salary are tightly correlated (around 0.9), which is expected. More interesting is that tech employment, innovation index, startup density, and degree attainment all move together with GDP, pointing to a knowledge-economy dimension that ties education to prosperity. Manufacturing employment sits on the other side, negatively correlated with most of those variables, suggesting a structural divide between industrial and knowledge-economy cities.

Green space and renewable energy are relatively independent from the economic block, hinting at a separate environmental dimension. Density, transit score, and commute time also cluster together, as expected for denser urban environments.

Taken together, these patterns suggest the 20 variables can be summarised by a small number of underlying dimensions, which is exactly what motivates the PCA step.

**The aim of the technical analysis** is to compress the variable space into interpretable components and then identify distinct city types using clustering.

---

## Technical Part

### Preprocessing

Two preprocessing steps were applied before any modelling.

`population`, `area_km2`, `density_per_km2`, and `startup_density_per_100k` were log-transformed using `np.log1p` to address the right skew identified above. Without this, a handful of very large cities dominate distance calculations simply because of their scale, which would push the clustering towards identifying "big vs small" rather than capturing more meaningful structural differences. After log-transforming, all 20 features were standardised to zero mean and unit variance using `StandardScaler`, since both PCA and distance-based clustering are sensitive to the scale of variables.

### Principal Component Analysis

PCA was applied to the scaled, log-transformed data. The scree plot shows that the first two components account for 55.5% and 28.7% of variance respectively, for a combined 84.2%. The first five together explain 93.5%.

[FIGURE PLACEHOLDER: PCA scree plot — third plot block in q2workspace_v2.py]

This rapid drop-off confirms the 20 variables contain a lot of redundancy, and a low-dimensional structure captures most of the relevant variation.

**PC1 (55.5% variance)** loads strongly on housing cost, degree attainment, startup density, GDP, innovation index, and tech employment. This is a wealth and knowledge-economy axis: cities with high PC1 scores are affluent, highly educated, and innovation-driven.

**PC2 (28.7% variance)** loads positively on renewable energy and green space, and negatively on density, population, transit score, and commute time. This captures an urban density vs greenness axis: cities scoring high on PC2 tend to be spread out and environmentally oriented, while those scoring low are dense metropolitan centres.

[FIGURE PLACEHOLDER: PCA loadings heatmap for PC1 and PC2 — fourth plot block in q2workspace_v2.py]

For clustering, 5 components were retained (93.5% variance). For visualisation, only the first two were used.

### Hierarchical Clustering

Ward's linkage was applied to the 5-component PCA space. Ward's method merges clusters in the way that minimises the increase in total within-cluster variance at each step, which tends to produce compact, similarly sized groups — a sensible default when no strong prior structure is assumed.

[FIGURE PLACEHOLDER: dendrogram — fifth plot block in q2workspace_v2.py]

The dendrogram shows a clear division into 4 main branches at a cut height of around 9. This guided the choice of k = 4 for the other methods.

### K-Means Clustering

K-means was run for k = 2 to 10 with 20 random initialisations each. Both the elbow in inertia and the silhouette scores were used to guide the choice of k.

[FIGURE PLACEHOLDER: elbow and silhouette plots — sixth plot block in q2workspace_v2.py]

Silhouette scores peak at k = 3 (0.557), then at k = 4 (0.535) and k = 6 (0.549). Despite k = 3 having the highest silhouette, k = 4 was chosen because it is consistent with the dendrogram and because the four-cluster solution produces more interpretable and distinguishable urban profiles. The drop from k = 3 to k = 4 is small (0.022). The final model used 30 initialisations with a fixed seed for reproducibility.

### DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) was also applied. Unlike the previous two methods, it does not require a pre-specified number of clusters and can flag low-density points as noise. The k-distance plot (k = 5) was used to select `eps = 1.8`, with `min_samples = 5`.

[FIGURE PLACEHOLDER: k-distance plot — seventh plot block in q2workspace_v2.py]

DBSCAN found 2 clusters and 4 noise points: a large cluster of 139 cities and a smaller one of 37. It essentially separates the dense megacities from the rest but does not distinguish the economic or environmental sub-groups that k-means picks up. This is not surprising: DBSCAN works best when clusters differ in density in the feature space, whereas here the four city types are separated more by their mean characteristics than by how tightly packed they are.

### Comparison of clustering methods

[FIGURE PLACEHOLDER: side-by-side PCA scatter for all three methods — eighth plot block in q2workspace_v2.py]

The PCA scatter shows that hierarchical clustering and k-means produce very similar partitions, with the four groups sitting in nearly the same positions in PC1-PC2 space. DBSCAN's two-group solution broadly overlaps with the megacity cluster versus the rest. The agreement between hierarchical and k-means gives more confidence in the 4-cluster solution.

### Cluster Profiles

[FIGURE PLACEHOLDER: standardised feature means heatmap — ninth plot block in q2workspace_v2.py]

The heatmap of standardised cluster means reveals four coherent urban archetypes:

**Cluster 0 — Knowledge Economy Hubs (36 cities).** Distinctively high GDP per capita (+1.05 SD), average salary (+1.07), degree attainment (+1.18), tech employment (+1.43), innovation index (+1.29), and startup density (+1.35). Unemployment is below average (-0.86) and housing is expensive (+0.62). These are prosperous, education-intensive cities driven by knowledge industries.

**Cluster 1 — Smaller, Green, Low-Density Cities (68 cities).** The largest cluster. Low transit scores (-1.05), low density (-0.75), short commutes (-1.02), above-average green space (+0.89) and renewable energy (+0.75). Economic indicators are broadly around average. These are smaller, spread-out cities with decent environmental quality but limited urban infrastructure.

**Cluster 2 — Dense Metropolitan Areas (38 cities).** By far the highest population (+1.73) and density (+1.65), with high transit scores (+1.32), housing costs (+1.40), diversity index (+1.44), and long commutes (+1.31). Below-average green space and air quality reflect the typical tradeoffs of dense urbanisation.

**Cluster 3 — Industrial or Economically Struggling Cities (38 cities).** Strongly defined by high unemployment (+1.73), heavy manufacturing employment (+1.83), older median age (+1.56), and poor air quality (+1.39). GDP, salaries, education, and tech employment are all well below average. These cities appear to follow an older industrial economic model with less diversified employment.

---

## Technical Reflections

### How many clusters appear to be present in the data?

The evidence points to somewhere between 3 and 4 clusters. The dendrogram shows a clear 4-branch structure when the tree is cut at a height of around 9, with each branch corresponding to a distinct group of cities. The k-means silhouette scores peak at k = 3 (0.557) and remain fairly stable at k = 4 (0.535), and the elbow in inertia also flattens around k = 3–4 with no single sharp inflection point.

At k = 3, the industrial cities (Cluster 3) and the green/suburban cities (Cluster 1) merge into one broad "non-metropolitan, non-knowledge-economy" group, which is really just defined by what those cities are not. k = 4 separates them, giving a more complete picture. So while the statistical criteria slightly favour k = 3, k = 4 tells a more meaningful story and is supported by the dendrogram. The two readings are not contradictory — k = 3 captures the coarser structure, k = 4 adds one layer of interpretable granularity.

### Should clustering be performed in the original feature space or the reduced one?

The 20 variables contain substantial redundancy — the correlation matrix shows tight blocks around the economic (GDP, salary, tech, innovation) and urban form (density, transit, commute) dimensions. Applying k-means directly to all 20 features effectively over-weights the knowledge-economy dimension, because it is represented by several highly correlated variables that each pull distances in the same direction. PCA redistributes this into orthogonal components and removes the double-counting, making distances in the reduced space more balanced and interpretable.

As a check, k-means was also run on the original standardised data without PCA reduction. The cluster profiles were broadly similar, but the groups showed more overlap in the PC1-PC2 projection and the silhouette score was slightly lower. The key practical benefit of working in PCA space here is that the geometry of the clustering becomes more honest: each retained component captures a genuinely independent dimension of variation, so distances reflect actual structural differences rather than coincidental co-linearity.

### How do the results differ across clustering methods, and are there any anomalies?

Hierarchical clustering (Ward) and k-means partition the cities in almost identical ways. Both identify the same four archetypes in nearly the same positions in PCA space, which is reassuring given that they approach the problem from different principles — one by agglomerative merging, the other by centroid optimisation.

DBSCAN diverges significantly, returning only 2 clusters and flagging 4 cities as noise. The noise points are worth noting: these 4 cities do not fall within any dense region of the 5-component PCA space, meaning they sit in low-density gaps between the main groups. They likely represent transitional cases — cities that do not fit cleanly into any archetype, perhaps undergoing economic restructuring or combining features from two distinct types. Their existence is a reminder that even in a clean synthetic dataset, not every observation maps neatly onto a cluster.

The more notable anomaly is DBSCAN recovering only 2 clusters when the other methods clearly identify 4. This is not a failure of DBSCAN — it is working as intended. It reflects the geometry of the data: the four city types are separated by their centroid positions in PCA space, not by differences in local density. DBSCAN's core assumption is that clusters are regions of high density separated by lower-density gaps, which does not match the structure here. This makes it a useful diagnostic: when DBSCAN and k-means disagree strongly, it tells you something about the shape of the clusters.

### How sensitive are the clustering results to preprocessing choices?

Three preprocessing decisions were made and each has material consequences.

*Log-transformation* is the most impactful choice. Without log-transforming the four right-skewed variables, PCA's first component effectively becomes a city-size axis. The few very large cities pull so strongly on variance that nearly all structure is explained by PC1 alone, and the clustering reduces to separating large cities from small ones rather than capturing economic or environmental differences. The log transform compresses these tails and allows the other dimensions to emerge.

*Standardisation* is also essential. Without it, variables measured in large units (GDP in USD, population in millions) dominate distance calculations regardless of their actual informativeness. Even after log-transforming, the variables remain on different scales, so standardisation is needed to ensure each feature contributes equally.

*Number of PCA components retained* has less impact than the above two. Testing with 3 components (86% variance) gives similar cluster profiles with slightly lower within-cluster coherence. Testing with 7 components produces no material change. The 4-cluster structure is robust to reasonable variation in how many components are kept, which suggests the signal is largely captured in the first 5.

### Do different clustering algorithms produce similar groupings?

Hierarchical clustering and k-means converge on very similar partitions. Visually, the four groups occupy nearly identical regions of PCA space under both methods, and inspecting the cluster membership shows high overlap. This agreement is meaningful precisely because the two methods have different failure modes and make different assumptions about cluster shape. Their convergence strengthens the case that the 4-cluster structure reflects something real in the data rather than being an artefact of one particular algorithm.

DBSCAN produces a coarser result, but this should be read as a structural comment rather than a disagreement. It is telling us that the clusters here are not density-based — they are centroid-based — and k-means and hierarchical methods are the right tools for that geometry.

### Do the clusters represent meaningful patterns, or do they reflect design assumptions in the data?

The four clusters correspond to recognisable urban archetypes and their profiles are internally coherent. However, since the data are synthetic, it is worth asking whether the analysis is recovering genuine structure or simply the structure that was built into the data generation process.

A few observations suggest the latter plays some role. The cluster separation in PCA space is quite clean — the four groups are clearly delineated with limited overlap in the scatter plot. In real-world urban data, cities rarely fall this neatly into discrete categories; gradients and transitional cases are far more common. The correlations within the economic block (GDP, salary, tech, innovation) are also unusually tight and consistent, more so than you would typically find in real data where city-specific histories and policies introduce noise.

That said, the cluster profiles are analytically meaningful even if they were designed in. The variables do align around interpretable urban dimensions — economic sophistication, population density, environmental orientation — and the archetypes they form correspond to city types that planners and urban economists would recognise. The honest caveat is that the clarity of the result is partly a product of the synthetic data rather than something guaranteed to replicate in messier real-world datasets. Applying the same pipeline to real city data would be a useful test of whether similar archetypes emerge, or whether the clean four-cluster structure dissolves into more of a continuum.

---

## Conclusions

### Non-technical summary

The analysis groups 180 cities into four types that correspond to recognisable urban patterns. The largest group consists of smaller towns with green environments and decent quality of life. A second group captures wealthy, tech-driven cities. A third covers large, dense, and diverse metropolitan centres. The fourth identifies cities with an industrial legacy and signs of economic strain: high unemployment, older populations, and poor air quality. These groupings came out consistently across two independent clustering methods, which suggests they reflect real structure in the data rather than being an artefact of any one algorithm.

### Limitations and extensions

The data are synthetic, which means the four profiles — while coherent and interpretable — may not generalise to real urban systems where history, policy, and geography all introduce complexity. The analysis is also purely descriptive: it identifies which cities are similar but says nothing about why they ended up that way. Possible extensions include soft clustering via Gaussian Mixture Models for probabilistic cluster membership, incorporating temporal data to track whether cities shift between archetypes over time, or applying the same pipeline to real-world urban datasets to test whether similar structures emerge in practice.
