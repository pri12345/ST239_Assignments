# ST239 Mindmap - Student ID 5558899 — Leaf Content Reference

---

## Branch 1 — Supervised Learning
*learn f(X) → Y from labelled data*

### Sub-branch: Regression *(continuous Y)*

| Node | Key equation | Fitting / procedure |
|------|-------------|---------------------|
| **[A] Simple Linear Regression** | Y = β₀ + β₁X + ε | OLS: minimise RSS, closed-form solution; β̂₁ = Cov(X,Y) / Var(X) |
| **[B] Multiple Linear Regression** | β̂ = (XᵀX)⁻¹XᵀY (matrix form) | OLS, p predictors, dummy encoding for categoricals; model selection via Adj. R², AIC/BIC, stepwise, CV |

### Sub-branch: Classification *(discrete Y)*

| Node | Key equation | Fitting / procedure |
|------|-------------|---------------------|
| **[C] Logistic Regression** | log(p/1−p) = Xβ → sigmoid | MLE (Bernoulli likelihood), iterative (Newton-Raphson); exp(βⱼ) = odds ratio |
| **[D] Decision Trees** | Split on Gini = Σpₖ(1−pₖ) (clf) / RSS (reg) | Recursive partitioning; pruning controls overfitting; leaf = majority class / mean |
| **[E] Random Forest** | Variance ↓ via averaging decorrelated trees | B trees on bootstrap samples, m random features per split; importance via MDI |
| **[E'] XGBoost** | F_t(x) = F_{t-1}(x) + η·h_t(x) | Boosting: sequential trees each correcting residuals; hyperparams: η, depth, n_estimators |
| **[SVM] Support Vector Machine** | min ‖w‖² s.t. yᵢ(wᵀxᵢ+b) ≥ 1 | Soft margin C, kernel trick (RBF K = exp(−γ‖x−x′‖²)), QP solver |

---

## Branch 2 — Unsupervised Learning
*find hidden structure, no labels*

### Sub-branch: Dimensionality Reduction

| Node | Key equation | Fitting / procedure |
|------|-------------|---------------------|
| **[F] Principal Component Analysis** | Σvᵢ = λᵢvᵢ (eigendecomposition of covariance matrix) | Standardise → eigendecompose → project onto top-k PCs; choose k via scree plot / cumulative variance ≥ 80% |

### Sub-branch: Clustering

| Node | Key equation / idea | Fitting / procedure |
|------|---------------------|---------------------|
| **[G] Hierarchical Clustering** | Dendrogram via agglomerative merge | Ward linkage: minimise within-cluster variance at each merge; cut at chosen height → k clusters |
| **[H] K-Means** | WCSS = Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖² | Lloyd's: assign points → update centroids → repeat; choose k via elbow plot + silhouette score |
| **[I] DBSCAN** | Dense regions separated by low-density areas | Params: ε (neighbourhood radius), min_samples; point types: core / border / noise (−1) |

---

## Branch 3 — Statistical Foundations
*cross-cutting tools, appear across all branches*

### Sub-branch: Parameter Estimation

| Node | Key equation | When it applies | Notes |
|------|-------------|-----------------|-------|
| **[J] OLS** | β̂ = (XᵀX)⁻¹XᵀY | SLR / MLR | Minimises RSS — closed-form solution |
| **[K] MLE** | θ̂ = argmax Σ log f(yᵢ \| θ) | Logistic Regression, any probabilistic model | Iterative (Newton-Raphson), requires distributional assumption |

### Sub-branch: Model Evaluation & Selection

| Node | Key idea | Metrics / tools |
|------|----------|-----------------|
| **[L] In-sample Criteria** | R², Adj. R² (penalise extra predictors), AIC = −2 log L + 2k | BIC uses log(n)·k — stricter for large n; lower = better |
| **[M] Cross-Validation & Metrics** | K-fold CV: train on k−1 folds, test on 1, rotate | Accuracy, F1, AUC-ROC; stratified CV for imbalanced data |

---

## Cross-branch connections (dotted arrows on map)

| Connection | Explanation |
|-----------|-------------|
| OLS = MLE (under Gaussian errors) | Both coincide when residuals are normally distributed |
| Logistic Regression ↔ MLE | LogReg is fitted via MLE with a Bernoulli likelihood, not OLS |
| PCA → Clustering | PCA is often used as a preprocessing step before K-Means / DBSCAN |
| Decision Trees → RF & XGBoost | Both ensemble methods build directly on decision trees |
| AIC/BIC ↔ MLE | AIC and BIC are defined in terms of the log-likelihood from MLE |
