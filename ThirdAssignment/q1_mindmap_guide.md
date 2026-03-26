# Q1 — MindMap Guide: ST239

---

## Suggested Caption

> "The map divides ST239 methods into three main branches: Supervised Learning (labelled data, predict Y), Unsupervised Learning (find hidden structure), and Statistical Foundations (cross-cutting tools for estimation and evaluation). Within Supervised Learning, a further split separates methods for continuous outputs (Regression) from methods for discrete outputs (Classification). Leaves represent individual methods, each labelled with a key equation or principle, its fitting procedure, and its interpretability level."

---

## Tree Structure

```
                          ┌───────────────────────────────────┐
                          │         ST239: Statistical        │
                          │         & ML Methods              │
                          └──────────────┬────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
   ┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
   │  Supervised         │   │  Unsupervised        │   │  Statistical         │
   │  Learning           │   │  Learning            │   │  Foundations         │
   └──────┬──────────────┘   └──────┬──────────────┘   └──────┬───────────────┘
          │                         │                          │
     ┌────┴─────┐            ┌──────┴──────┐           ┌──────┴──────┐
     ▼          ▼            ▼             ▼            ▼             ▼
 Regression  Classif.    Dim. Red.     Clustering   Estimation  Evaluation
     │          │            │             │            │             │
   [A][B]   [C][D][E]       [F]        [G][H][I]     [J][K]      [L][M]
             [E']
```

---

## Branch 1 — Supervised Learning

**What unites these methods**: learn a mapping f(X) → Y from labelled training data. The goal is prediction (regression) or classification.

---

### Sub-branch: Regression (continuous Y)

#### [A] Simple Linear Regression
- **Key idea**: model a linear relationship between one predictor X and a continuous response Y.
- **Key equation**: `Y = β₀ + β₁X + ε`
- **Fitting**: OLS — minimise RSS = Σ(yᵢ − ŷᵢ)². Closed-form: `β̂₁ = Cov(X,Y)/Var(X)`.
- **Assumptions (LINE)**: Linearity, Independence, Normality of residuals, Equal variance (homoscedasticity).
- **Interpretability**: ★ White box — β₁ = expected change in Y per unit increase in X.
- **Probabilistic?**: No (OLS is frequentist); yes under Gaussian error assumption (connects to MLE).

#### [B] Multiple Linear Regression
- **Key idea**: extends SLR to p predictors. `Y = Xβ + ε` in matrix form.
- **Key equation**: `β̂ = (XᵀX)⁻¹XᵀY` — requires no perfect multicollinearity.
- **Fitting**: OLS (same criterion as SLR, closed-form solution in matrix form).
- **Extra consideration**: dummy variable encoding for categoricals (k−1 dummies for k levels); model selection via Adj. R², AIC/BIC, stepwise, CV.
- **Interpretability**: ★ White box — each βⱼ = ceteris paribus effect.
- **Shared connection with [A]**: same assumptions and fitting philosophy; SLR is a special case.

---

### Sub-branch: Classification (discrete Y)

#### [C] Logistic Regression
- **Key idea**: binary classification — model log-odds of Y=1 as a linear function of X.
- **Key equation**: `log(p/1−p) = Xβ`  →  `p = 1/(1+e^{−Xβ})` (sigmoid function).
- **Fitting**: MLE — maximise log-likelihood (no closed form; iterative, e.g. Newton-Raphson). Not OLS.
- **Interpretation**: exp(βⱼ) = odds ratio for a one-unit increase in Xⱼ.
- **Interpretability**: ★ White box — coefficients directly interpretable as log-odds.
- **Probabilistic?**: Yes — outputs calibrated probabilities.
- **Shared connection**: OLS = MLE under Gaussian errors [→ Branch 3, node J]; logistic regression uses MLE but with Bernoulli likelihood.

#### [D] Decision Trees *(Regression & Classification)*
- **Key idea**: recursively partition the feature space into rectangular regions. At each node, pick the split that maximally reduces impurity.
- **Classification split criterion**: Gini = Σ pₖ(1−pₖ). Leaf prediction = majority class.
- **Regression split criterion**: minimise RSS within children. Leaf prediction = mean.
- **Overfitting control**: max depth, min samples per leaf, pruning.
- **Interpretability**: ★ White box (shallow) → ★★ Grey box (deep). Can draw the tree.
- **Probabilistic?**: No.
- **Note**: symmetrical connection to Ensemble Methods — RF and XGBoost build on trees.

#### [E] Random Forest
- **Key idea**: train B trees on bootstrap samples; at each split consider only m random features. Average predictions (regression) or majority vote (classification).
- **Key concept**: variance reduction via averaging decorrelated trees. Low bias per tree, reduced variance in ensemble.
- **Fitting**: bootstrap + random feature subsets. No closed form; ensemble of OLS/Gini minimisations.
- **Feature importance**: Mean Decrease in Impurity (MDI).
- **Interpretability**: ★★ Grey box.
- **Probabilistic?**: No (but outputs class proportions as probability estimates).
- **Scalability**: O(B × n log n × √p per split) — good, parallelisable.

#### [E'] XGBoost *(could sit alongside RF, or as a sub-leaf of Ensemble Methods)*
- **Key idea**: boosting — build trees sequentially, each correcting the residuals of the current ensemble. `F_t(x) = F_{t-1}(x) + η·hₜ(x)`.
- **Key difference from RF**: sequential not parallel; trees are shallow; regularisation (L1/L2) in objective.
- **Hyperparameters**: learning rate η, depth, n_estimators, subsample.
- **Interpretability**: ★★ Grey box (SHAP values can help).
- **Probabilistic?**: No.
- **Scalability**: generally fast, state-of-the-art on tabular data.

#### [F_sv] Support Vector Machines *(SVM)*
- **Key idea**: find the hyperplane that maximises the margin between two classes. Only the support vectors (nearest points) determine the boundary.
- **Key equation (hard margin)**: minimise `‖w‖²` subject to `yᵢ(wᵀxᵢ+b) ≥ 1`.
- **Soft margin**: C controls penalty for misclassification (large C = tight fit).
- **Kernel trick**: map features to higher-dimensional space implicitly via K(xᵢ,xⱼ). RBF kernel: `K = exp(−γ‖x−x'‖²)`.
- **Fitting**: quadratic programming (convex, no local minima). Numerical optimisation, O(n²–n³).
- **Interpretability**: ★★★ Black box (non-linear kernels).
- **Probabilistic?**: No (but Platt scaling can convert scores to probabilities).
- **Scalability**: slow for large n.

> **Note on placement**: SVM fits under Classification. Add a note that it can also handle regression (SVR).

---

## Branch 2 — Unsupervised Learning

**What unites these methods**: no labelled Y — the task is to find structure, reduce complexity, or group observations.

---

### Sub-branch: Dimensionality Reduction

#### [F] Principal Component Analysis (PCA)
- **Key idea**: find new orthogonal axes (PCs) that capture maximum variance. Project data into a lower-dimensional space.
- **Key equation**: eigendecomposition of covariance matrix Σ: `Σvᵢ = λᵢvᵢ`. PCs = eigenvectors; variance explained = eigenvalues.
- **Procedure**: standardise → compute Σ → eigendecompose → project onto top-k PCs.
- **Choosing k**: scree plot elbow; cumulative variance threshold (e.g. 80%); Kaiser criterion (λ > 1).
- **Loadings**: coefficients of original variables in each PC — high loading = strong contribution.
- **Interpretability**: ★★ Grey box — PCs are linear combinations, less intuitive than original features.
- **Probabilistic?**: No.
- **Connection**: pre-processing step feeding into Clustering below.

---

### Sub-branch: Clustering

#### [G] Hierarchical Clustering
- **Key idea**: build a dendrogram by iteratively merging (agglomerative) or splitting (divisive) clusters.
- **Linkage**: Ward (minimise within-cluster variance on merge) / Complete / Single.
- **Output**: cut the dendrogram at chosen height → k clusters. No need to specify k in advance.
- **Interpretability**: ★ White box — dendrogram is fully readable.
- **Scalability**: O(n² log n) — does not scale to very large n.
- **Probabilistic?**: No.

#### [H] K-Means
- **Key idea**: iteratively assign each point to the nearest centroid, then recompute centroids. Minimise WCSS.
- **Key equation**: `WCSS = Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖²`
- **Procedure (Lloyd's)**: initialise k centroids → assign → update → repeat until convergence.
- **Choosing k**: elbow plot (WCSS vs k); silhouette score `s(i) = (b−a)/max(a,b)`.
- **Assumptions**: spherical, similarly-sized clusters; sensitive to outliers and initialisation.
- **Interpretability**: ★ White box (cluster centres are interpretable).
- **Scalability**: O(nkTd) per iteration — fast, scales well.
- **Probabilistic?**: No.

#### [I] DBSCAN
- **Key idea**: density-based clustering — clusters are dense regions separated by low-density areas. Can find arbitrary shapes and labels noise.
- **Key parameters**: ε (neighbourhood radius), min_samples (density threshold).
- **Point types**: core / border / noise (labelled −1).
- **Choosing ε**: k-distance plot — look for the elbow in sorted distance to k-th nearest neighbour.
- **Pros**: no need to specify k; detects outliers naturally; handles non-spherical shapes.
- **Cons**: struggles with varying density; sensitive to ε.
- **Interpretability**: ★ White box (cluster memberships are outputs; their meaning must be interpreted).
- **Scalability**: worse than K-Means, but approximations exist.
- **Probabilistic?**: No.

---

## Branch 3 — Statistical Foundations

**What unites these**: cross-cutting concepts for fitting models and evaluating them — they appear across both supervised and unsupervised methods.

---

### Sub-branch: Parameter Estimation

#### [J] Ordinary Least Squares (OLS)
- **Key idea**: estimate model parameters by minimising the sum of squared residuals.
- **Key equation**: `β̂ = (XᵀX)⁻¹XᵀY` (closed-form, linear models only).
- **Connection**: OLS = MLE under Gaussian errors — the two frameworks coincide here.
- **When it applies**: SLR, MLR.

#### [K] Maximum Likelihood Estimation (MLE)
- **Key idea**: choose parameters θ that make the observed data most probable under the model.
- **Key equation**: `θ̂ = argmax Σ log f(yᵢ | θ)` (maximise log-likelihood).
- **Connections**: OLS is MLE under Gaussian errors; Logistic Regression uses MLE with Bernoulli likelihood.
- **Fitting**: iterative numerical optimisation (no closed form in general).
- **Probabilistic?**: Yes — requires a distributional assumption on the data.

---

### Sub-branch: Model Evaluation & Selection

#### [L] In-sample Criteria (R², Adj. R², AIC/BIC)
- **R²**: proportion of variance explained = `1 − RSS/TSS`. Always increases with more predictors (misleading).
- **Adjusted R²**: penalises for additional predictors: `1 − (RSS/(n−p−1)) / (TSS/(n−1))`.
- **AIC**: `−2 log L + 2k` — penalises complexity; lower = better. BIC uses log(n)·k (stricter for large n).
- **Use case**: comparing models fit to the same dataset; AIC/BIC allow non-nested comparison.

#### [M] Cross-validation & Classification Metrics
- **Cross-validation**: estimate out-of-sample performance by training/testing on different folds. K-fold: split data into k folds, train on k−1, test on 1, rotate. Stratified CV preserves class balance.
- **Classification metrics**:
  - Accuracy = (TP+TN)/n — misleading under imbalance.
  - Precision = TP/(TP+FP); Recall = TP/(TP+FN).
  - F1 = 2·P·R/(P+R) — harmonic mean, balances both.
  - AUC-ROC: area under TPR vs FPR curve at all thresholds; 0.5 = random, 1 = perfect.

---

## Symmetrical / Shared Connections to Note on the Map

These are links that cut across branches — worth drawing as dotted arrows or shared tags:

| Connection | Note |
|---|---|
| OLS ↔ MLE | OLS minimises RSS; under Gaussian errors this is identical to MLE |
| Decision Trees → RF & XGBoost | Ensemble methods build directly on decision trees |
| PCA → Clustering | PCA is often used as a pre-processing step before K-Means / DBSCAN |
| Logistic Regression ↔ MLE | Fitted via MLE (Bernoulli likelihood), not OLS |
| AIC/BIC ↔ MLE | AIC/BIC are defined in terms of the log-likelihood |
| K-Means ↔ Hierarchical | Both require specifying structure; Ward linkage ≈ K-Means objective |

---

## Interpretability Quick Reference (for leaf tags)

| Method | Box colour |
|---|---|
| SLR / MLR / Logistic Regression | White |
| Decision Trees (shallow) | White |
| Decision Trees (deep) | Grey |
| Random Forest / XGBoost | Grey |
| SVM (non-linear kernel) | Black |
| PCA | Grey |
| Clustering methods | White (outputs) |

---

## Drawing Tips

- **Layout suggestion**: Supervised Learning on the left, Unsupervised on the right, Statistical Foundations at the bottom (or top). This way the PCA → Clustering and MLE → Logistic Regression arrows flow naturally.
- **Leaf tags**: small rounded rectangle for each leaf containing: (1) key equation or one-sentence principle; (2) fitting method; (3) interpretability colour.
- **Shared branches**: draw OLS and MLE on the Statistical Foundations branch, then add dotted lines back to SLR/MLR and Logistic Regression respectively — avoids repeating them as leaves elsewhere.
- **Caption**: include a 3–4 sentence caption below the map explaining the top-level organisation logic (supervised vs unsupervised, and the foundations branch).
