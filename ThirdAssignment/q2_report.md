# Question 2: Exploring Urban Profiles Using Dimensionality Reduction and Clustering

---

## Introduction

The dataset `Q2_cities.csv` contains 180 fictitious cities described by 20 numerical variables spanning five broad themes: demographics (population, area, density, median age), economic performance (GDP per capita, average salary, unemployment rate, share with a bachelor's degree or higher), infrastructure (transit score, housing cost index, commute time), environmental characteristics (green space, air quality, renewable energy share), and sectoral composition (tech, manufacturing, tourism, innovation indices, and startup density).

The central objective is to determine whether cities naturally organise into groups with coherent urban profiles — and, if so, to characterise what distinguishes those groups. To achieve this, we combine dimensionality reduction via Principal Component Analysis (PCA) with three clustering algorithms: hierarchical clustering, k-means, and DBSCAN.

---

## Exploratory Data Analysis

### Summary statistics and distributions

The dataset is complete — no missing values — and all variables are numerical, which simplifies preprocessing. Summary statistics reveal substantial heterogeneity across the 180 cities. Population ranges from roughly 80,000 to 14 million, a spread of nearly two orders of magnitude, and area shows a similarly wide range (35–980 km²). Most economic and index variables (GDP, transit score, innovation) are more symmetrically distributed, with standard deviations roughly 15–25% of their means.

Histograms (produced by the first plot block in `q2workspace.py`) confirm that `population`, `area_km2`, `density_per_km2`, and `startup_density_per_100k` are strongly right-skewed, with most cities clustering at lower values and a long tail of much larger ones. The remaining variables are approximately unimodal and roughly bell-shaped, with occasional mild skew.

### Pairwise relationships

The correlation heatmap (second plot in `q2workspace.py`) reveals several clear patterns:

- **Economic cluster**: `gdp_per_capita_usd` and `avg_salary_usd` are tightly correlated (≈0.9), as expected. `bachelors_or_higher_pct`, `tech_employment_pct`, `innovation_index_0_100`, `startup_density_per_100k`, and `housing_cost_index` all co-vary positively with GDP — capturing a wealth/knowledge-economy dimension.
- **Urban form**: `density_per_km2`, `transit_score_0_100`, and `commute_time_min` move together, reflecting the infrastructure profile of denser cities.
- **Counter-indicator**: `manufacturing_employment_pct` correlates negatively with most economic and innovation indicators, suggesting a structural divide between knowledge-economy and industrial cities.
- **Environmental variables**: `green_space_m2_per_capita` and `renewable_energy_pct` are relatively independent of the economic block, hinting at a separate dimension related to environmental orientation.

These patterns already suggest that city variation is driven by a small number of latent axes — motivating dimensionality reduction.

**The aim of the technical analysis** is to compress the 20-variable space into interpretable components and then identify distinct city types using unsupervised clustering methods.

---

## Technical Part

### Preprocessing

Two preprocessing steps were applied before any modelling.

**Log-transformation of skewed variables.** `population`, `area_km2`, `density_per_km2`, and `startup_density_per_100k` were replaced by their natural logarithm (using `np.log1p`). This compresses the long right tails and prevents large-magnitude cities from dominating distance calculations purely through scale. The decision was based on the histogram inspection above.

**Standardisation.** After log-transforming, all 20 features were standardised to zero mean and unit variance using `StandardScaler`. This is essential for both PCA and distance-based clustering: without it, variables measured in USD or people/km² would dominate components and distances simply because of their larger numerical ranges.

It is worth noting the sensitivity of results to these choices. Running the analysis without log-transforming the skewed variables shifts the PCA axes and tends to create one dominant cluster consisting of the few very large megacities while flattening structure elsewhere. Standardisation alone is not sufficient to address this.

### Principal Component Analysis

PCA was applied to the standardised, log-transformed data. The scree plot (third figure in `q2workspace.py`) shows that the first two principal components capture **55.5%** and **28.7%** of total variance respectively, for a combined **84.2%**. Adding three more components reaches 93.5%. This rapid drop-off confirms that the 20 variables contain substantial redundancy and can be well-summarised by a low-dimensional structure.

**Interpretation of PC1 and PC2** (loadings heatmap, fourth figure):

- **PC1 (55.5% variance)** loads positively and nearly equally on `housing_cost_index`, `bachelors_or_higher_pct`, `startup_density_per_100k`, `gdp_per_capita_usd`, `innovation_index_0_100`, and `tech_employment_pct`. This is the *wealth and knowledge-economy* axis: cities with a high PC1 score are affluent, highly educated, innovation-driven, and expensive to live in.

- **PC2 (28.7% variance)** loads positively on `renewable_energy_pct` and `green_space_m2_per_capita` and negatively on `density_per_km2`, `population`, `transit_score_0_100`, and `commute_time_min`. This is the *urban density versus greenness* axis: cities scoring high on PC2 are spread out, green, and environmentally oriented; those scoring low are dense, transit-heavy megacities.

For clustering we retained **5 principal components** (93.5% of variance) to preserve more of the data structure. For visualisation we project onto the first two.

### Hierarchical Clustering

Ward's linkage was applied to the 5-component PCA space. Ward's method minimises the increase in total within-cluster variance at each merge step, which tends to produce compact and comparably sized clusters — a sensible default when no strong prior structure is known.

The dendrogram (fifth figure) shows a clear division into **4 main branches** at a cut height of approximately 9, with two of those branches each splitting cleanly into two sub-groups. We therefore proceed with **k = 4**.

### K-Means Clustering

K-means was run for k = 2 to 10, with 20 random initialisations each (elbow and silhouette plots, sixth figure). The elbow in inertia is visible around k = 3–4. Silhouette scores peak at **k = 3 (0.557)**, then again at k = 4 (0.535) and k = 6 (0.549).

While k = 3 has the highest silhouette, **k = 4** was chosen for the final model because: (i) it is consistent with the dendrogram, (ii) it produces four meaningfully distinct urban archetypes rather than three coarser groups, and (iii) the silhouette drop from k = 3 to k = 4 is small (0.022). The final model used 30 initialisations with a fixed random seed for reproducibility.

### DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) was also applied. Unlike k-means and hierarchical methods it requires no predetermined number of clusters and can identify arbitrarily shaped clusters while labelling low-density regions as noise. The k-distance plot (seventh figure, k = 5) suggested an `eps` of approximately 1.8, and `min_samples = 5`.

DBSCAN found **2 clusters** and **4 noise points**: a large cluster of 139 cities and a smaller one of 37. This result is less informative than the partitional methods — it essentially separates the dense megacities from the rest without distinguishing the economic or environmental sub-groups that k-means captures. This is not surprising: DBSCAN performs best when clusters have meaningfully different densities in feature space, whereas here the four city types are more separated by direction (mean) than by density.

### Comparison of clustering methods

The side-by-side PCA scatter (eighth figure) shows that hierarchical clustering and k-means produce very similar partitions — the four groups are largely co-located in PC1–PC2 space. DBSCAN's two-group solution overlaps broadly with the megacity cluster (Cluster 2) versus the rest. Agreement between hierarchical and k-means strengthens confidence in the 4-cluster solution.

### Cluster Profiles

The heatmap of standardised feature means (ninth figure, `q2workspace.py`) reveals four coherent urban archetypes:

**Cluster 0 — Knowledge Economy Hubs (36 cities)**
Cities like Highford and Newsprings. Distinctively high GDP per capita (+1.05 SD), average salary (+1.07), bachelors degree share (+1.18), tech employment (+1.43), innovation index (+1.29), and startup density (+1.35). Housing costs are elevated (+0.62) and unemployment is below average (−0.86). These are prosperous, education-intensive cities that attract skilled workers and innovation-led industries.

**Cluster 1 — Smaller, Green, Low-Density Cities (68 cities)**
The largest cluster — cities like Longstead and Pinehurst. Characterised by low transit scores (−1.05), low density (−0.75), and short commute times (−1.02), combined with above-average green space (+0.89) and renewable energy (+0.75). Economic indicators are around the average. These are the smaller, more spread-out cities with good environmental quality but limited urban infrastructure.

**Cluster 2 — Dense Metropolitan Areas (38 cities)**
Cities like Cedarton and Brightsprings. By far the highest population (+1.73), density (+1.65), transit scores (+1.32), housing costs (+1.40), and diversity (+1.44). Commute times are long (+1.31). Green space and air quality are below average, reflecting the tradeoffs of dense urbanisation. These are the large metropolitan centres.

**Cluster 3 — Industrial or Economically Struggling Cities (38 cities)**
Cities like Kettleside and Mapleton. Strongly defined by high unemployment (+1.73), high manufacturing employment (+1.83), older median age (+1.56), and poor air quality (+1.39). GDP, salaries, education, and tech employment are all well below average. These cities appear to represent an older, industrial economic model with less diversified employment and greater social vulnerability.

---

## Conclusions

### Non-technical summary

The analysis groups 180 cities into four types that correspond to recognisable urban patterns. The largest group consists of smaller towns with good quality of life and green environments. A second group captures wealthy, tech-driven cities. A third group covers large, dense, and diverse metropolitan centres. The fourth identifies cities with a heavy industrial legacy and signs of economic strain: high unemployment, older populations, and poorer air quality. These groupings are consistent across two independent clustering methods, suggesting they reflect real structure in the data rather than artefacts of the algorithm.

### Technical discussion

- **Sensitivity to preprocessing**: log-transforming skewed variables before scaling materially improves cluster separation by preventing a handful of very large cities from dominating the distance metric. Without it, PCA is effectively dominated by city size alone.
- **Method agreement**: hierarchical (Ward) and k-means produce closely aligned partitions (visible in the eighth figure), which is reassuring. DBSCAN's coarser result reflects its density-based geometry — it is not well-suited to clusters that differ primarily in their centroid locations.
- **Choice of k**: silhouette peaks at k = 3, suggesting that at a coarser level the main structural divide is between large dense metros, knowledge economy cities, and the rest. k = 4 adds interpretable granularity by separating green/suburban cities from industrial ones.
- **Limitations**: the data are synthetic, and the cluster profiles — while coherent — may not generalise to real urban systems where causal relationships and policy contexts matter. The analysis is also purely descriptive; it does not establish why cities belong to a given type or what interventions might shift them.
- **Potential extensions**: clustering in the original 20-dimensional space (rather than the PCA-reduced one) could be compared; soft clustering methods (e.g. Gaussian Mixture Models) would allow probabilistic cluster membership; and incorporating temporal data would enable tracking of cities moving between archetypes over time.
