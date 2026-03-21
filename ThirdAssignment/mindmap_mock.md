# ST239 MindMap — Topic Research Notes

---

## 1. Simple Linear Regression

**Model overview**: Models a linear relationship between one predictor X and a response Y:
`Y = β₀ + β₁X + ε`
β₀ is the intercept, β₁ is the slope, ε is the error term (noise).

**Assumptions (LINE)**:
- **L**inearity: the relationship between X and Y is linear
- **I**ndependence: observations are independent of each other
- **N**ormality: residuals are normally distributed
- **E**qual variance (homoscedasticity): residual variance is constant across fitted values

**Fitting criterion (OLS)**: Minimise the Residual Sum of Squares (RSS):
`RSS = Σ(yᵢ - ŷᵢ)²`
The closed-form solution for β₁ = Cov(X,Y) / Var(X), β₀ = ȳ - β₁x̄

**Interpretation**: β₁ = expected change in Y for a one-unit increase in X, holding everything else constant. Intercept β₀ = predicted Y when X=0 (may not be meaningful in context).

**Diagnostics**:
- Residuals vs Fitted plot (check linearity + homoscedasticity)
- Q-Q plot of residuals (check normality)
- Scale-Location plot
- R² = 1 - RSS/TSS: proportion of variance in Y explained by the model

**Interpretability**: White box — full transparency, coefficients directly interpretable.

---

## 2. Multiple Linear Regression

**Model overview**: Extends SLR to p predictors:
`Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε`

**Matrix notation**: `Y = Xβ + ε`
where Y is n×1, X is n×(p+1) design matrix (first column all 1s for intercept), β is (p+1)×1.

**OLS solution**: `β̂ = (XᵀX)⁻¹XᵀY` — requires XᵀX to be invertible (no perfect multicollinearity).

**Assumptions**: Same as SLR + no perfect multicollinearity among predictors.

**Prediction**: For a new observation x*, predicted value = x*ᵀβ̂. Can produce confidence intervals (for the mean response) and prediction intervals (for a single new observation — wider).

**Interpretability**: White box — each βⱼ = effect of Xⱼ on Y holding all other predictors constant (ceteris paribus).

---

## 3. Categorical Variables, Model Selection, R² and Adjusted R²

**Categorical variables**: Cannot enter a model as raw numbers — encoded as dummy variables.
- A categorical variable with k levels → k-1 binary dummies (one dropped as reference category to avoid the dummy variable trap / perfect multicollinearity).
- Interpretation: coefficient on a dummy = difference in predicted Y relative to the reference level.

**R²**: Proportion of variance explained: `R² = 1 - RSS/TSS`. Always increases (or stays the same) when more predictors are added — can be misleadingly optimistic.

**Adjusted R²**: Penalises for adding uninformative predictors:
`R²_adj = 1 - (RSS/(n-p-1)) / (TSS/(n-1))`
Increases only if the new predictor improves the model more than expected by chance.

**Model Selection**: Choosing the right set of predictors.
- **AIC / BIC**: information-criterion based — penalise model complexity, lower is better.
- **Stepwise selection**: forward, backward, or both — heuristic search through model space.
- **Cross-validation**: more principled, estimates out-of-sample predictive performance directly.
- Overfitting risk: a model with too many predictors fits noise, not signal.

---

## 4. Maximum Likelihood Estimation (MLE), Connection with OLS, Variable Significance, Log-Likelihood and AIC

**MLE principle**: Find parameter values θ that maximise the likelihood of observing the data:
`θ̂_MLE = argmax L(θ; data) = argmax Π f(yᵢ | θ)`
In practice, maximise the log-likelihood (log-L) — mathematically equivalent, numerically easier.

**Connection with OLS**: Under the assumption that ε ~ N(0, σ²), maximising the log-likelihood of a linear regression is equivalent to minimising RSS (OLS). So OLS = MLE under Gaussian errors.

**Variable significance**:
- **t-test**: H₀: βⱼ = 0. Test statistic = β̂ⱼ / SE(β̂ⱼ), compared against t-distribution with n-p-1 degrees of freedom.
- **p-value**: probability of observing a test statistic at least as extreme under H₀. Small p-value → evidence against H₀.
- **Confidence interval**: β̂ⱼ ± t* · SE(β̂ⱼ)

**Log-Likelihood**: `log L(θ) = Σ log f(yᵢ | θ)`. Higher is better (less negative). Used as the basis for comparing models.

**AIC (Akaike Information Criterion)**:
`AIC = -2 log L + 2k` where k = number of parameters.
Penalises complexity. Lower AIC = better model. Useful for comparing non-nested models.
BIC uses log(n)·k as the penalty (stronger penalty for large n).

---

## 5. Logistic Regression

**The problem**: Y is binary (0/1) — linear regression would predict values outside [0,1].

**Logistic / sigmoid function**: Maps any real number to (0,1):
`σ(z) = 1 / (1 + e^{-z})`

**Logit link**: Model the log-odds:
`log(p/(1-p)) = β₀ + β₁X₁ + ... + βₚXₚ`
so `p = σ(Xβ)`

**Model fitting**: No closed-form OLS solution. Parameters found by maximising the log-likelihood (via iterative numerical optimisation, e.g. Newton-Raphson / gradient descent).

**Parameter interpretation**:
- βⱼ = change in log-odds for a one-unit increase in Xⱼ.
- exp(βⱼ) = odds ratio: factor by which the odds of Y=1 multiply for a one-unit increase in Xⱼ.

**Classification**: Choose a threshold (default 0.5): predict Y=1 if p̂ > threshold, else Y=0.

**Performance metrics**:
- **Accuracy**: (TP + TN) / total — can be misleading for imbalanced classes.
- **Precision**: TP / (TP + FP) — of predicted positives, how many are actually positive?
- **Recall (Sensitivity)**: TP / (TP + FN) — of actual positives, how many did we catch?
- **F1**: 2 · (Precision · Recall) / (Precision + Recall) — harmonic mean, balances both.
- **Confusion matrix**: 2×2 table of TP, TN, FP, FN.
- **AUC-ROC**: Area under the ROC curve (TPR vs FPR at all thresholds) — threshold-independent measure of discrimination ability. 0.5 = random, 1.0 = perfect.

**Interpretability**: White box — coefficients are directly interpretable (as log-odds / odds ratios).

---

## 6. Decision Trees (Regression & Classification)

**Core idea**: Recursively partition the feature space into rectangular regions. At each node, choose the split (feature + threshold) that best reduces impurity / error.

**Classification trees**:
- **Gini impurity**: `G = Σ pₖ(1 - pₖ)` — measures how often a randomly chosen element would be misclassified. Lower = purer node.
- **Entropy (information gain)**: alternative splitting criterion.
- Leaf prediction: majority class in the leaf.

**Regression trees**:
- Split criterion: minimise RSS within child nodes.
- Leaf prediction: mean of training observations in that leaf.

**Stopping criteria** (to prevent overfitting):
- Maximum depth
- Minimum samples per leaf / per split
- Minimum impurity decrease threshold
- Pre-pruning (stop early) or post-pruning (grow full tree then cut back)

**Pros**: Highly interpretable (can draw the tree), handles non-linearities and interactions, no scaling needed.
**Cons**: High variance — small data changes can produce very different trees. Prone to overfitting if unconstrained.

**Interpretability**: White box (shallow) → grey box (deep trees become hard to follow).

---

## 7. Ensemble Methods — Random Forest & XGBoost

### Random Forest

**Key idea**: Build many decorrelated decision trees and average their predictions (bagging + random feature subsets).

**Procedure**:
1. Draw B bootstrap samples from training data.
2. For each sample, grow a deep decision tree — but at each split, only consider a random subset of m features (typically √p for classification, p/3 for regression).
3. Aggregate: majority vote (classification) or average (regression).

**Why it works**: Individual trees have high variance but low bias. Averaging reduces variance without increasing bias. The random feature subsets decorrelate the trees so the ensemble actually benefits.

**Feature importance**: Mean Decrease in Impurity (MDI) — how much each feature reduces impurity across all trees/splits.

**Pros**: Very strong out-of-the-box performance, robust to overfitting, handles high-dimensional data.
**Cons**: Less interpretable than a single tree, slower to train/predict for very large datasets.

**Interpretability**: Grey box.

---

### XGBoost (Extreme Gradient Boosting)

**Key idea**: Boosting — build trees sequentially, where each new tree corrects the residual errors of the current ensemble.

**Procedure**:
1. Start with a simple prediction (e.g. mean of Y).
2. Fit a tree to the residuals (negative gradient of the loss).
3. Add the tree to the ensemble with a learning rate η (shrinkage).
4. Repeat for T rounds.

**Key differences from Random Forest**:
- Sequential, not parallel.
- Each tree is typically shallow (stumps or depth 3-6).
- Learning rate η controls how much each tree contributes — smaller = more robust, needs more trees.
- Regularisation terms (L1/L2 on leaf weights) built into the objective.

**Pros**: Often achieves state-of-the-art performance on tabular data, handles missing values, highly customisable.
**Cons**: More hyperparameters to tune (learning rate, depth, n_estimators, subsample...), can overfit if not tuned carefully.

**Interpretability**: Grey box (SHAP values can help explain individual predictions).

---

## 8. Principal Component Analysis (PCA)

**Goal**: Reduce dimensionality by finding new orthogonal axes (principal components) that capture maximum variance in the data.

**Procedure**:
1. Standardise the data (mean 0, unit variance) — important: PCA is scale-sensitive.
2. Compute the covariance matrix Σ.
3. Eigendecomposition: find eigenvalues λ₁ ≥ λ₂ ≥ ... and eigenvectors v₁, v₂, ...
4. Project data onto the top k eigenvectors (principal components).

**Eigenvalues**: Each eigenvalue = variance explained by that PC. `Σvᵢ = λᵢvᵢ`

**Criteria for choosing k (number of components)**:
- **Explained variance threshold**: keep enough PCs to explain e.g. 80% of total variance.
- **Scree plot**: plot eigenvalues in order — look for the "elbow" where the curve flattens. Components before the elbow carry most of the signal.
- **Kaiser criterion**: keep PCs with eigenvalue > 1 (i.e. explains more variance than a single original variable).
- **Cross-validation**: treat k as a hyperparameter and select by predictive performance.

**Loadings**: The coefficients of each original variable in a PC. High absolute loading = that variable contributes strongly to that PC.

**Pros**: Reduces noise, removes multicollinearity, enables 2D/3D visualisation.
**Cons**: PCs are linear combinations — less interpretable than original features. Information is lost.

**Interpretability**: Grey box — the PCs themselves are harder to interpret than raw features.

---

## 9. Support Vector Machines (SVM)

**Core idea (linear, hard margin)**: Find the hyperplane that separates two classes with the maximum margin. The margin is the distance between the hyperplane and the nearest points of each class (the support vectors).

**Objective**: Maximise margin = 2/||w||, subject to correct classification.
Equivalently: minimise ||w||² subject to yᵢ(wᵀxᵢ + b) ≥ 1.

**Soft margin (C parameter)**: Allow some misclassifications (slack variables ξᵢ). C controls the trade-off: large C = penalise misclassification heavily (tighter fit), small C = wider margin, more violations tolerated.

**The kernel trick**: For non-linearly separable data, map features to a higher-dimensional space via a function φ(x). The kernel K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ) computes dot products in that space *without explicitly computing φ* — computationally efficient.

**Kernel types**:
- **Linear**: K(x, x') = xᵀx' — standard dot product, no transformation.
- **Polynomial**: K(x, x') = (γ xᵀx' + r)^d — captures polynomial interactions up to degree d.
- **Gaussian RBF (Radial Basis Function)**: K(x, x') = exp(-γ||x - x'||²) — very flexible, maps to infinite-dimensional space. Most commonly used for non-linear problems. γ controls the "reach" of each training point.
- **Sigmoid**: K(x, x') = tanh(γ xᵀx' + r) — analogous to a neural network activation.

**Hyperparameters**: C (margin/regularisation), and kernel-specific (γ for RBF, degree for polynomial).

**Pros**: Effective in high-dimensional spaces, memory-efficient (only support vectors matter), works well with a good kernel.
**Cons**: Slow on very large datasets (O(n²-n³) training), sensitive to feature scaling, hard to interpret, kernel choice is non-trivial.

**Interpretability**: Black box (especially with non-linear kernels).

---

## 10. Clustering — Overview

**Goal**: Unsupervised — find natural groupings in data without labels.

**Three main families**:

### Hierarchical Clustering
- Builds a tree of clusters (dendrogram).
- **Agglomerative** (bottom-up): start with each point as its own cluster, merge the two closest clusters at each step.
- **Divisive** (top-down): start with one cluster, recursively split.
- **Linkage criteria** (how to measure distance between clusters):
  - Single linkage: distance between nearest points
  - Complete linkage: distance between farthest points
  - Ward: minimise increase in total within-cluster variance on merge (tends to give compact, equal-sized clusters)
- **Dendrogram**: visualises the merge hierarchy. Cut at a chosen height → k clusters.
- **Pros**: No need to pre-specify k; the dendrogram reveals structure at multiple scales.
- **Cons**: O(n² log n) or O(n³) — doesn't scale to large n; merges are irreversible.

### Centroid-based (K-Means) — detailed in section 11

### Density-based (DBSCAN) — detailed in section 12

---

## 11. K-Means — Detailed

**Procedure (Lloyd's algorithm)**:
1. Initialise k centroids (randomly, or via k-means++ for smarter initialisation).
2. **Assignment step**: assign each point to the nearest centroid (Euclidean distance).
3. **Update step**: recompute each centroid as the mean of its assigned points.
4. Repeat steps 2-3 until assignments stop changing (convergence).

**Distance metric**: Euclidean distance (L2). This means K-Means implicitly assumes clusters are spherical and similarly sized.

**Convergence**: Guaranteed to converge (the objective — within-cluster sum of squares, WCSS — is non-increasing), but may converge to a local minimum. Multiple restarts (n_init) help.

**Objective**: Minimise WCSS = `Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²`

**Pros**: Simple, fast (O(nkTd) per iteration), scales well.
**Cons**:
- Must specify k in advance.
- Sensitive to initialisation (mitigated by k-means++).
- Assumes spherical, similarly-sized clusters — fails on elongated or irregular shapes.
- Sensitive to outliers (means are pulled by extreme values).
- Not deterministic (random init).

**Choosing k**:
- **Elbow method**: plot WCSS vs k — look for the "elbow" where adding another cluster gives diminishing returns.
- **Silhouette score**: `s(i) = (b(i) - a(i)) / max(a(i), b(i))` where a(i) = mean intra-cluster distance, b(i) = mean nearest-cluster distance. Ranges from -1 to 1; higher = better defined clusters. Average over all points.

---

## 12. DBSCAN — Detailed

**Procedure**:
1. For each point, count how many points fall within radius ε (the neighbourhood).
2. **Core points**: points with ≥ min_samples neighbours within ε.
3. **Border points**: within ε of a core point, but fewer than min_samples neighbours themselves.
4. **Noise points**: neither core nor border — labelled -1.
5. Clusters = connected components of core points (and their border points).

**Key concepts**:
- **ε (eps)**: neighbourhood radius. Too small → most points are noise. Too large → everything merges.
- **min_samples**: minimum points to be a core point. Higher = stricter, more noise labelled.

**Choosing hyperparameters**:
- **k-distance plot**: for each point, compute the distance to its k-th nearest neighbour (k = min_samples). Sort and plot. The "elbow" in this curve is a good estimate for ε.

**Pros**:
- Does not require specifying k in advance.
- Can find arbitrarily shaped clusters.
- Naturally identifies and labels noise/outliers.

**Cons**:
- Struggles when clusters have very different densities (a single ε can't work for all).
- Sensitive to ε and min_samples — requires careful tuning.
- Doesn't scale as well as K-Means for very large datasets (though approximations exist).

**Interpretability**: The cluster labels are the output — the clusters themselves must be interpreted by examining their members.

---

*End of topic notes — awaiting review before building the node-by-node guide.*
