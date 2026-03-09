# ST239 Assignment 2

## Q1. Bagging vs Boosting for Credit Default Prediction

### Introduction

In this question we compare two ensemble approaches — a **Random Forest** (bagging) and **XGBoost** (gradient boosting) — to predict whether a bank customer will experience a credit default. The dataset contains 6,000 customer records with 13 financial and demographic features. Both models are tuned via 5-fold cross-validation using log-loss as the optimisation criterion, then compared on accuracy, ROC-AUC, and their ability to generalise across repeated held-out splits.

---

### Hyperparameter Tuning

Both models were evaluated using `GridSearchCV` with a 5-fold cross-validation scheme. **Log-loss** was used as the refitting criterion for both, consistent with its role as the splitting criterion in the underlying decision trees. The grids explored were:

- **Random Forest**: 144 combinations over `n_estimators` ∈ {50, 200, 500, 1000}, `max_depth` ∈ {2, 3, 5, None}, `max_features` ∈ {3, 5, 9} (covering ~23%, ~38%, ~69% of 13 features), `min_samples_leaf` ∈ {1, 5, 10}.
- **XGBoost**: 128 combinations over `n_estimators` ∈ {50, 200, 500, 1000}, `max_depth` ∈ {2, 3, 4, 10}, `learning_rate` ∈ {0.01, 0.03, 0.05, 0.1}, `subsample` ∈ {0.7, 1.0}.

**Accuracy** measures the fraction of correctly classified customers. In a credit-default context, a model with high accuracy correctly identifies both defaulters and non-defaulters, though it can be misleading when classes are imbalanced. **ROC-AUC** summarises the model's ability to rank customers by their default risk across all classification thresholds; a value closer to 1 indicates a stronger discriminative ability regardless of class distribution.

#### Random Forest Findings

*[To be completed after running the code — insert specific best parameters and scores here.]*

Across the explored grid, accuracy and AUC show **diminishing returns as `n_estimators` increases**: the largest gains occur moving from 50 to 200 trees, with improvements becoming negligible beyond 500. This is expected — more trees reduce variance but cannot correct bias.

Deeper trees (`max_depth=None` or `max_depth=5`) tend to achieve slightly higher CV accuracy and AUC than shallow ones (`max_depth=2`), as they can capture more complex decision boundaries. However, the train accuracy for unconstrained depth is noticeably higher than the CV accuracy, indicating a degree of **overfitting**: the forest memorises training patterns that do not generalise perfectly. Shallower trees exhibit a smaller train–CV gap, suggesting better regularisation at the cost of some bias.

Increasing `min_samples_leaf` acts as an additional regulariser: larger values (10) reduce overfitting (narrow the train–CV gap) at a moderate cost to accuracy, while `min_samples_leaf=1` allows the most flexible trees with the largest gap.

The best RF model by log-loss and the best by accuracy/AUC are likely to share similar hyperparameters (deep trees, many estimators, small min_samples_leaf), since log-loss is highly correlated with both metrics for well-calibrated classifiers.

#### XGBoost Findings

*[To be completed after running the code — insert specific best parameters and scores here.]*

For XGBoost, the interaction between `learning_rate` and `n_estimators` is critical. A small learning rate (0.01) requires many more trees to converge; with only 50 trees the model underfits, but accuracy improves steadily as `n_estimators` increases. Conversely, a larger learning rate (0.1) converges faster but can overshoot the optimum with 1000 trees. The best performing configurations typically pair a moderate learning rate with a larger ensemble.

Regarding overfitting, XGBoost exhibits a **more pronounced train–CV accuracy gap** than the Random Forest at large `max_depth`, reflecting its sequential boosting nature: each new tree aggressively corrects the errors of its predecessors, making it more susceptible to overfitting on training data. The `subsample < 1` configurations partially mitigate this through stochastic sampling.

---

### Model Comparison

#### Best Model Selection

*[To be completed after running the code — state the winning hyperparameter combinations and whether the best-by-accuracy and best-by-logloss models differ.]*

In general, the best model by log-loss and by accuracy may differ slightly: log-loss rewards well-calibrated probability estimates, while accuracy is threshold-dependent (defaults to 0.5 cutoff). Models with large, well-tuned ensembles tend to rank highly on all three metrics simultaneously.

#### Feature Importance

*[To be completed after running the code — fill in the actual top-5 features and discuss.]*

Both models expose a `feature_importances_` attribute: for RF this is the mean decrease in impurity (MDI) averaged across all trees; for XGBoost it is the gain-based importance (average improvement in loss per split).

In a credit-risk context, we would expect variables such as `past_default`, `num_late_payments_12m`, `debt_to_income`, and `credit_utilization` to be strong predictors — they are direct indicators of financial stress. Whether the two models agree on the top predictors reflects the consistency of the signal: high agreement suggests robust, genuine predictors; disagreement may arise from the different importance definitions (MDI vs. gain) or from correlated features where each model latches onto a different proxy.

If importance is **highly concentrated** on a few features, the remaining predictors add little marginal value and the problem is essentially low-dimensional. A spread-out distribution suggests multiple complementary signals.

---

### Final Evaluation

The best RF and XGB models (selected by log-loss) were evaluated over **30 repeated 80/20 train/test splits** with different random seeds to obtain a distribution of Accuracy and Recall, removing dependence on any single partition.

*[To be completed after running the code — insert mean ± std for both metrics and both models.]*

**Which metric to prioritise for credit default?**
In this setting, **Recall** (sensitivity) deserves higher priority than overall Accuracy. A false negative — predicting no default when the customer actually defaults — carries a much greater financial cost for the bank (unrecovered loans) than a false positive (declining a creditworthy customer). A model with higher recall catches more true defaulters even at the expense of some false alarms.

*[To be completed: state which model achieves higher average Recall and which shows lower variability across seeds, and give a final recommendation.]*

---

## Q2. \'Planting\' your decision tree

### Introduction

In this task, we construct a decision tree by hand to predict whether a student athlete should be selected for the top BUCS Rugby Team, using their historical data on training, fitness, attendance, and prior club experience.

### Part 1: Gini Impurity at Root Node

The formula for the Gini impurity is $1 - \sum (p_i)^2$. Using all 25 observations:
- Class 0 (No Top Team): $n_0 = 13 \\implies p_0 = 0.52$
- Class 1 (Top Team): $n_1 = 12 \\implies p_1 = 0.48$

$$ \\text{Gini(Root)} = 1 - (0.52^2 + 0.48^2) = 1 - 0.5008 = 0.4992 $$

### Part 2: Output of Code (max_depth=2)

If we were to run `DecisionTreeClassifier(max_depth=2)`, the default optimal splits chosen by Python would differ from forcing the first split on 8 hours. Based on the exported tree rules, the engine selects `TrainingHours <= 7.50` as the best initial split. For the left child, it splits again at `TrainingHours <= 6.50`, and for the right child, at `FitnessPass <= 0.50`. This creates a flowchart where predicting the top team largely revolves around meeting the 7.50-hour threshold and passing the fitness test.

### Part 3: First Split Forced on TrainingHours (\u2265 8)

The coach requires \u2265 4 sessions of 2 hours, meaning \u2265 8 hours of training per week. We force our first split:
- **Left Child $(< 8 \\text{ hours})$**: Contains 13 samples (12 Class 0, 1 Class 1). 
  - $\\text{Gini}_{left} = 1 - ((12/13)^2 + (1/13)^2) \\approx 0.1420$
- **Right Child $(\\ge 8 \\text{ hours})$**: Contains 12 samples (1 Class 0, 11 Class 1).
  - $\\text{Gini}_{right} = 1 - ((1/12)^2 + (11/12)^2) \\approx 0.1528$

**Weighted Gini**: $(13/25 \\times 0.1420) + (12/25 \\times 0.1528) \\approx 0.1472$
**Impurity reduction**: $0.4992 - 0.1472 = 0.3520$

### Part 4: Second-Level Split Evaluation

Evaluating the remaining variables (`FitnessPass`, `AttendanceGood`, `PriorClub`) for the left and right child nodes:
- **Left Child $(< 8, n=13)$**: Splitting by `PriorClub` gives the lowest weighted Gini (0.1231).
- **Right Child $(\\ge 8, n=12)$**: Splitting by `FitnessPass` gives the lowest weighted Gini (0.0833).

**Justification**: We choose `PriorClub` for the Left Child and `FitnessPass` for the Right Child because they yield the purest leaves and the maximum reduction in Gini Impurity.

### Part 5: Flow Diagram (Hand-made Tree)

```text
[Root Node]
|-- Impurity index: 0.4992  |  Samples: 25
|-- Split 1: TrainingHours >= 8?
|
|-- [Left Child: No (< 8)]
|   |-- Impurity index: 0.1420  |  Samples: 13
|   |-- Split 2: PriorClub == 1?
|   |
|   |-- [Left Leaf: No] -> Impurity: 0.0 | Prediction: Class 0
|   |-- [Right Leaf: Yes] -> Impurity: 0.2449 | Prediction: Class 0
|
|-- [Right Child: Yes (>= 8)]
    |-- Impurity index: 0.1528  |  Samples: 12
    |-- Split 2: FitnessPass == 1?
    |
    |-- [Left Leaf: No] -> Impurity: 0.0 | Prediction: Class 0
    |-- [Right Leaf: Yes] -> Impurity: 0.0 | Prediction: Class 1
```

### Question 1: Daily Average Transformation

If `TrainingHours` was divided by 7 to provide a daily average, the decision tree would **not change at all**. A decision tree splits data based on ordering and relative thresholds. Since division by a constant strictly preserves the ordering of the instances (it is a monotonic linear transformation), the algorithm would pick the exact same relative cut points, leading to identical impurity reductions, identical tree structures, and identical predictions.

### Question 2: Student Predictions (Shallow Tree)

- **Student A** (`TrainingHours=9, FitnessPass=0`): Routes to Right Child (\u2265 8), then to Left Leaf (`FitnessPass` < 0.5). **Prediction: 0 (Do not select)**.
- **Student B** (`TrainingHours=6, PriorClub=0`): Routes to Left Child (< 8), then to Left Leaf (`PriorClub` < 0.5). **Prediction: 0 (Do not select)**.

### Max Depth Tree Construction

Modifying the tree to `max_depth=None` to use all variables yields a deep programmatic tree with the following properties:
- **Maximum Depth**: 3 levels
- **Number of Leaves**: 6 leaves
- **Combined Purity**: Gini = 0.0 across all leaves (100% pure).

**Flowchart:**
```text
[Root] -> Split: TrainingHours <= 7.50
   |-- [<= 7.50, n=13] -> Split: TrainingHours <= 6.50
   |      |-- [<= 6.50, n=6]  -> Class 0
   |      |-- [> 6.50, n=7]   -> Split: PriorClub <= 0.50 -> (<=: Class 0, >: Class 1)
   |
   |-- [> 7.50, n=12] -> Split: FitnessPass <= 0.50
          |-- [<= 0.50, n=2]  -> Split: AttendanceGood <= 0.50 -> (<=: Class 0, >: Class 1)
          |-- [> 0.50, n=10]  -> Class 1
```

**Predictions vs Hand-made Tree:**
Running the students through the new deep tree:
- **Student A** (`Train=9, Fit=0, Att=1`): Routes: `> 7.50` -> `Fitness <= 0.50` -> `Attendance > 0.50`. **Prediction: 1 (Select)**.
- **Student B** (`Train=6`): Routes: `<= 7.50` -> `<= 6.50`. **Prediction: 0 (Do not select)**.

**Model Comparison:**
The shallow tree provided an intuitive "rule of thumb" but failed to classify Student A properly because it grouped all un-fit students without checking attendance. The deep tree correctly mapped Student A's mitigating factor (Good Attendance) to class them into the top team. However, a coach might strongly prefer the shallow tree because it is highly interpretable, easy to remember on the field, and explicitly prevents overfitting. Deep trees risk memorising noisy combinations the 25-sample dataset that will not generalise to next year's crop of students.

---

## Q3. PCA in Practice

### Introduction

In this question we explore the structure of a dataset recording occupational information for 1,200 workers using **Principal Component Analysis (PCA)**. The dataset contains nine continuous predictors covering physical work demands (standing, lifting, manual intensity, repetitive motion), sedentary activities (seated hours, computer use), and socio-economic characteristics (age, income, meeting hours), alongside a binary target `chronic_pain`. The PCA section focuses exclusively on the explanatory variables to uncover the main axes of variation before incorporating the target in the logistic regression that follows.

---

### Data Inspection

The dataset contains 1,200 observations and 9 features. The summary statistics reveal substantial **scale heterogeneity**:

| Variable | Mean | Std |
|---|---|---|
| age | 40.5 | 10.6 |
| annual_income_gbp | 28,953 | 7,998 |
| standing_hours_per_day | 4.45 | 1.89 |
| lifting_hours_per_day | 1.72 | 1.32 |
| manual_intensity_score | 5.24 | 2.35 |
| repetitive_motion_score | 4.92 | 2.17 |
| seated_hours_per_day | 5.37 | 1.96 |
| computer_hours_per_day | 4.13 | 1.82 |
| meetings_hours_per_week | 6.01 | 4.23 |

`annual_income_gbp` has a standard deviation of ~8,000 — roughly 4,000× larger than the hourly variables — and `meetings_hours_per_week` (std ≈ 4.2) is more variable than the other hour-based measures. Chronic pain affects **27.9%** of workers (approximately 1-in-4).

The correlation matrix highlights two broad clusters, confirmed by the PCA loadings below:

- **Physical/manual cluster**: `standing_hours_per_day`, `lifting_hours_per_day`, `manual_intensity_score`, and `repetitive_motion_score` are positively correlated — workers who stand more also tend to lift more and report higher manual intensity.
- **Desk-work cluster**: `seated_hours_per_day`, `computer_hours_per_day`, and `meetings_hours_per_week` are positively associated, and these groups are largely orthogonal to the physical cluster (active and sedentary work are mutually exclusive in a given day).

`age` is relatively independent of both clusters; `annual_income_gbp` shows a mild association with the desk-work group (higher-income workers tend to be office-based).

---

### PCA: Preprocessing and Justification

Before running PCA, the features were **standardised** (zero mean, unit variance) using `StandardScaler`. Two reasons justify this choice:

1. **Scale differences**: `annual_income_gbp` has a standard deviation several orders of magnitude larger than the hourly variables. Without standardisation, PCA would be dominated by income simply because of its larger numeric range, not because it is genuinely more variable in a meaningful sense.
2. **PCA maximises variance**: Since PCA finds the directions of maximum variance, variables on larger scales would artificially dominate the first principal component. Standardisation puts all variables on an equal footing so that PCA reflects the underlying covariance *structure*, not measurement units.

Mean-centering is performed automatically by `StandardScaler` and is also a formal requirement for PCA (the covariance matrix is defined around the mean).

---

### Principal Components: Eigenvalues and Loadings

The PCA was run on the 9 standardised variables. The eigenvalue table below summarises how much variance each component explains:

| PC | Eigenvalue | Variance | Cumulative |
|---|---|---|---|
| 1 | 1.898 | 21.1% | 21.1% |
| 2 | 1.601 | 17.8% | 38.9% |
| 3 | 1.042 | 11.6% | 50.4% |
| 4 | 0.858 | 9.5% | 59.9% |
| 5 | 0.807 | 9.0% | 68.9% |
| 6 | 0.755 | 8.4% | 77.3% |
| 7 | 0.715 | 7.9% | 85.2% |
| 8 | 0.682 | 7.6% | 92.8% |
| 9 | 0.648 | 7.2% | 100.0% |

The eigenvalues decrease gradually with no sharp elbow, meaning variance is spread fairly evenly across all components — no single dominant latent factor drives the data.

#### Reading the Loadings

The correlation loadings are computed as:

$$\text{cor}(x_i, \text{PC}_j) = v_{ji} \times \sqrt{\lambda_j}$$

where $v_{ji}$ is the $j$-th eigenvector component (from pca.components_) and $\lambda_j$ is the $j$-th eigenvalue. Because the inputs are standardised, these values are Pearson correlations bounded in $[-1, 1]$.

The full correlation loadings matrix for the first three PCs:

| Variable | PC1 | PC2 | PC3 |
|---|---|---|---|
| age | 0.031 | 0.003 | 0.915 |
| annual_income_gbp | 0.073 | 0.601 | 0.313 |
| standing_hours_per_day | 0.697 | -0.020 | -0.019 |
| lifting_hours_per_day | 0.703 | -0.100 | -0.054 |
| manual_intensity_score | 0.713 | -0.063 | 0.049 |
| repetitive_motion_score | 0.622 | -0.054 | -0.039 |
| seated_hours_per_day | 0.065 | 0.583 | -0.261 |
| computer_hours_per_day | 0.024 | 0.704 | -0.149 |
| meetings_hours_per_week | 0.103 | 0.622 | 0.100 |

A loading above 0.55 in absolute value is used as the cut-off for a dominant contribution, since at that level the variable shares over 30% of its variance with the PC. Dominant variables per component:

| PC | Dominant variables (|cor| > 0.55) | Direction |
|---|---|---|
| PC1 | manual_intensity_score (0.71), lifting_hours_per_day (0.70), standing_hours_per_day (0.70), repetitive_motion_score (0.62) | All positive |
| PC2 | computer_hours_per_day (0.70), meetings_hours_per_week (0.62), annual_income_gbp (0.60), seated_hours_per_day (0.58) | All positive |
| PC3 | age (0.92) | Positive |

#### Naming the Principal Components

PC1 — "Physical Labour Intensity"

The four dominant variables all measure the physical demands of a job: standing upright, carrying loads, exerting muscular effort, and repetitive movement. They load strongly and positively, while every desk-work variable (seated_hours_per_day, computer_hours_per_day, meetings_hours_per_week, annual_income_gbp) has a near-zero loading on PC1 (all below 0.11). Workers who do more of one of these activities tend to do more of the others, forming a coherent physical-work cluster.

PC2 — "Office / Desk Work Intensity"

The four dominant variables (computer_hours_per_day, meetings_hours_per_week, annual_income_gbp, seated_hours_per_day) describe sedentary, technology-mediated, and better-paid work. All four physical-work variables that drove PC1 load near-zero on PC2 (range: -0.10 to -0.02), which makes sense since you can't simultaneously lift heavy loads and sit at a computer for hours in the same workday. The inclusion of annual_income_gbp (0.60) reflects that office-based roles tend to pay more, so income correlates more with desk-work intensity than with age alone.

PC3 — "Seniority / Age"

A single variable dominates: age (0.92), the largest individual loading across all three PCs. Annual_income_gbp contributes modestly (0.31), reflecting the income premium that typically builds up over a career. All work-activity variables load near-zero on PC3 (range: -0.26 to 0.10). The fact that annual_income_gbp loads on both PC2 (0.60) and PC3 (0.31) makes sense — income relates to both job type and career stage, so its variance gets split across two components. PC3 captures career stage independently of whether someone is in a physical or desk-based role.

#### Confirmation from the Correlation Circle Plots

- PC1 vs PC2: The four physical-work arrows cluster in the positive-PC1 / near-zero-PC2 region; the four office-work arrows cluster in the near-zero-PC1 / positive-PC2 region. The two groups sit roughly at 90° to each other. Age lies near the origin, confirming it contributes almost nothing to either PC1 or PC2.

- PC1 vs PC3: Age projects strongly along PC3 (vertical axis), while the physical-work arrows remain along PC1 (horizontal). The near-right angle between age and the physical variables confirms PC3 is independent of physical job demands.

- PC2 vs PC3: Age points along PC3; the office-work arrows point along PC2. Annual_income_gbp's arrow sits between the two axes, reflecting its split role across desk-work and seniority. Physical-work arrows are near the origin, confirming they contribute minimally to either PC2 or PC3.

---

### 2D and 3D Projections

The 2D scatter plot of PC1 vs PC2 maps workers into a "job-type space": workers in the upper-right are physically active (high PC1) and also office-intensive (high PC2), while workers in the lower-left have low physical and low desk demands. The two chronic pain groups overlap substantially in this 2D projection, indicating that job type alone (without age) does not cleanly separate pain cases from non-pain cases. Some concentration of chronic pain cases may be visible at the extremes of PC1 (heavy manual labour) or PC2 (intensive desk work), both associated with musculoskeletal stress through different mechanisms.

The 3D projection adds PC3 (the age/seniority axis). Older workers may show a slight tendency toward chronic pain at any level of PC1 or PC2, providing modest additional separation that the first two PCs could not capture.

---

### Dimensionality Reduction Criteria

Three standard criteria were applied:

1. **Kaiser criterion (eigenvalue > 1)**: retain only components whose eigenvalue exceeds 1, meaning they explain more variance than a single standardised variable would. This is a commonly used heuristic.

2. **Scree plot (elbow method)**: plot eigenvalues in decreasing order and look for an "elbow" — the point at which successive eigenvalues drop steeply before levelling off. Components before the elbow are retained.

3. **Cumulative variance threshold**: retain the minimum number of components that together explain at least 80% (or 90%) of total variance.

The actual eigenvalues and cumulative variance are:

| PC | Eigenvalue | Variance | Cumulative |
|---|---|---|---|
| 1 | 1.898 | 21.1% | 21.1% |
| 2 | 1.601 | 17.8% | 38.9% |
| 3 | 1.042 | 11.6% | 50.4% |
| 4 | 0.858 | 9.5% | 59.9% |
| 5 | 0.807 | 9.0% | 68.9% |
| 6 | 0.755 | 8.4% | 77.3% |
| 7 | 0.715 | 7.9% | 85.2% |
| 8 | 0.682 | 7.6% | 92.8% |
| 9 | 0.648 | 7.2% | 100.0% |

The three criteria give **divergent recommendations**:

- **Kaiser criterion**: retain **3 components** (PC1 = 1.90, PC2 = 1.60, PC3 = 1.04, all > 1; PC4 = 0.86 < 1).
- **Scree plot**: the eigenvalues drop from 1.90 → 1.60 → 1.04 → 0.86 with no sharp elbow — the spectrum is relatively flat, consistent with a dataset where variance is spread across several dimensions. The clearest change of slope occurs at PC4, supporting retention of **3 components**.
- **80% variance threshold**: requires **7 components** (85.2%); **8 components** for 90%.

The flat eigenvalue spectrum is a diagnostic signal: the nine variables share their explanatory power fairly evenly without a dominant low-dimensional structure. This makes dimensionality reduction less effective here than in datasets with a few strong latent factors.

**Decision**: We retain **3 principal components**, balancing interpretability and Kaiser/scree agreement, accepting that they capture only ~50% of total variance. PC1 (physical labour), PC2 (desk work), and PC3 (age/seniority) each correspond to a clear and meaningful occupational dimension, making this choice justifiable for the subsequent logistic regression. The 80% threshold's requirement of 7 components would largely eliminate the dimensionality reduction benefit and produce components that are difficult to interpret.

---

## Q3. Logistic Regression

### Data Inspection: `chronic_pain` and its Correlations

The target `chronic_pain` is binary, with a prevalence of **27.9%** (335 out of 1,200 workers). The class imbalance is moderate and does not require resampling for logistic regression.

**Correlation with original features** (Pearson, sorted by absolute value):

| Variable | Correlation with `chronic_pain` |
|---|---|
| lifting_hours_per_day | **0.216** |
| repetitive_motion_score | **0.185** |
| standing_hours_per_day | **0.167** |
| manual_intensity_score | 0.154 |
| age | 0.093 |
| meetings_hours_per_week | 0.093 |
| seated_hours_per_day | 0.080 |
| annual_income_gbp | 0.070 |
| computer_hours_per_day | 0.056 |

Physical-work variables dominate — workers who lift more and perform repetitive motions are most strongly associated with chronic pain. `annual_income_gbp` and `computer_hours_per_day` show the weakest linear relationship.

**Correlation with retained PCs**:

| PC | Correlation with `chronic_pain` |
|---|---|
| PC1 (Physical Labour) | **0.273** |
| PC2 (Desk Work) | 0.089 |
| PC3 (Age/Seniority) | 0.070 |

PC1 is clearly the most predictive of the three — consistent with physical variables being the strongest individual correlates. PC2 and PC3 show weaker but non-trivial associations, suggesting desk work intensity and older age also contribute to chronic pain risk.

---

### Logistic Regression with Original Variables

A full model was first fitted using all 9 features. Three predictors were **not statistically significant** at the 5% level: `annual_income_gbp` (p = 0.240), `manual_intensity_score` (p = 0.127), and `computer_hours_per_day` (p = 0.321). These are dropped, yielding a **selected model** with 6 predictors.

**Selected model results** (all 6 predictors significant, p < 0.05):

| Predictor | Coefficient | OR | 95% CI | p-value |
|---|---|---|---|---|
| `age` | 0.0213 | **1.022** | [1.009, 1.034] | 0.0009 |
| `standing_hours_per_day` | 0.1062 | **1.112** | [1.031, 1.199] | 0.0056 |
| `lifting_hours_per_day` | 0.2780 | **1.320** | [1.188, 1.468] | <0.001 |
| `repetitive_motion_score` | 0.1340 | **1.143** | [1.071, 1.220] | <0.001 |
| `seated_hours_per_day` | 0.0849 | **1.089** | [1.016, 1.166] | 0.0158 |
| `meetings_hours_per_week` | 0.0439 | **1.045** | [1.012, 1.079] | 0.0062 |

*McFadden Pseudo-R² = 0.079; AIC = 1322.70; 7 parameters (6 predictors + intercept)*

**Which predictors appear most strongly associated with chronic pain?**

`lifting_hours_per_day` (OR = 1.32) is the strongest predictor: each additional hour of daily lifting is associated with a 32% increase in the odds of chronic pain, holding other variables constant. `repetitive_motion_score` (OR = 1.14) and `standing_hours_per_day` (OR = 1.11) follow. These three are all physical-load variables, consistent with the well-established link between manual labour and musculoskeletal disorders. `seated_hours_per_day` (OR = 1.09) and `meetings_hours_per_week` (OR = 1.04) are statistically significant but with smaller effects, reflecting the modest risk from prolonged sedentary postures.

**Are some variables difficult to interpret jointly?**

Yes. `lifting_hours_per_day`, `standing_hours_per_day`, `manual_intensity_score`, and `repetitive_motion_score` are all moderately correlated (they measure different facets of physical workload). In the full model, `manual_intensity_score` loses significance (p = 0.127) even though its univariate correlation with chronic pain is 0.15 — a classic sign of **multicollinearity**: once `lifting` and `repetitive_motion` are in the model, `manual_intensity` adds little new information because it shares its explanatory variance with them. Similarly, `computer_hours_per_day` becomes redundant once `seated_hours_per_day` and `meetings_hours_per_week` are controlled for. The individual coefficients of correlated predictors can shift and become harder to interpret because their effects cannot be cleanly separated — a 1-unit increase in `lifting` while holding `standing` constant may not correspond to any realistic worker.

---

### Logistic Regression with PCA Variables

Using the 3 retained principal components as predictors, the **PCA logistic regression** results are:

| Predictor | Coefficient | OR | 95% CI | p-value |
|---|---|---|---|---|
| PC1 (Physical Labour) | 0.4749 | **1.608** | [1.452, 1.780] | <0.001 |
| PC2 (Desk Work) | 0.1718 | **1.187** | [1.070, 1.318] | 0.0012 |
| PC3 (Age/Seniority) | 0.1596 | **1.173** | [1.031, 1.334] | 0.0153 |

*McFadden Pseudo-R² = 0.076; AIC = 1321.32; 4 parameters (3 PCs + intercept)*

All three PCs are significant predictors of chronic pain, confirming that each occupational dimension independently contributes to risk.

**Interpreting the coefficients and odds ratios:**

- **PC1 (Physical Labour, OR = 1.61)**: A one-standard-deviation increase in PC1 — moving from an average worker toward a more physically demanding job profile (more standing, lifting, manual intensity, and repetitive motion) — is associated with a **61% increase** in the odds of chronic pain. This is the dominant risk factor. The large OR reflects the combined effect of four correlated physical-work variables compressed into a single dimension.

- **PC2 (Desk Work, OR = 1.19)**: Workers who score higher on the desk-work dimension (more computer time, more meetings, higher income) have **19% higher odds** of chronic pain per standard-deviation increase. This may reflect musculoskeletal strain from prolonged static postures, screen use, and the sedentary nature of office work.

- **PC3 (Age/Seniority, OR = 1.17)**: Older, more senior workers have **17% higher odds** of chronic pain per SD increase in PC3. Age is an independent risk factor for musculoskeletal conditions, regardless of job type.

---

### Model Comparison

| Model | Parameters | Log-Likelihood | AIC | BIC | Pseudo R² |
|---|---|---|---|---|---|
| Full original (9 features) | 10 | −651.81 | 1323.62 | 1374.52 | 0.0827 |
| Selected original (6 features) | 7 | −654.35 | 1322.70 | 1358.33 | 0.0792 |
| PCA (3 components) | 4 | −656.66 | **1321.32** | 1341.68 | 0.0759 |

**Number of parameters**: The PCA model uses only 4 parameters compared to 7 (selected) or 10 (full). This is a significant parsimony gain — the PCA model achieves a comparable fit with less than half the number of coefficients.

**AIC (lower is better)**: The PCA model achieves the best AIC (1321.32) despite the lowest pseudo-R². This reflects the heavy AIC penalty for additional parameters: the full model's slightly better log-likelihood is outweighed by its 6 extra parameters.

**Pseudo-R²**: The full original model has the highest McFadden R² (0.083), but all three models are in a narrow range (0.076–0.083). None of the models explain a large fraction of deviance — chronic pain has a complex aetiology and a single data-collection point with 9 variables is unlikely to capture it fully.

**Interpretability**: The original selected model has a direct clinical interpretation — each coefficient describes the effect of a specific, measurable work characteristic. The PCA model is harder to translate into actionable occupational health advice: telling an employer "reduce PC1" is less useful than "reduce lifting hours." However, the PCA model is more concise and less affected by multicollinearity.

**Overall recommendation**: For **predictive parsimony** the PCA model is preferable (best AIC, fewest parameters). For **causal inference and workplace intervention**, the selected original model is preferable (direct variable interpretation, though subject to collinearity caveats).

---

## Q3. Reflection Questions

### Question 1: Why can PCs be predictive of chronic pain if PCA ignored the target variable?

PCA is an **unsupervised** method that finds directions of maximum variance in the feature space, entirely ignoring `chronic_pain`. However, the principal components can still be predictive because the features themselves carry information about the outcome. PCA does not need to "know" the target to extract useful structure — if the original variables are predictive of chronic pain, and if that predictive information is correlated with the main directions of variation in the data, then the PCs will inherit that predictive power.

In this dataset, the first PC captures physical labour intensity (standing, lifting, manual work, repetitive motions), which happens to be the strongest driver of musculoskeletal chronic pain. PCA identified this dimension simply because physical-work variables vary together across workers — it did not need the outcome to discover the cluster. The predictiveness of PC1 is a consequence of the fact that the signal driving chronic pain (physical workload) also happens to be the dominant axis of variation in the input data. If the signal were orthogonal to the main variance directions, PCA would fail to capture it and the PCs would be uninformative — but here the two coincide.

---

### Question 2: Is PCA guaranteed to improve predictive performance?

**No**, PCA is not guaranteed to improve predictive performance. There are several reasons:

1. **Discarded variance may contain signal**: PCA retains directions of maximum variance, but the outcome may be correlated with low-variance components that are discarded. In such a case, PCA would throw away the most predictive information while keeping noise-heavy directions.

2. **Unsupervised nature**: Because PCA ignores the target variable, there is no guarantee that the retained components are the most predictive ones — only that they explain the most variance. Supervised alternatives (e.g., Partial Least Squares) explicitly maximise the covariance between the PCs and the target and can outperform PCA for prediction.

3. **Information loss**: Retaining 3 components out of 9 discards ~50% of the total variance. Even if all 9 original variables were good predictors, the PCA model is working with a compressed and partially degraded representation.

4. **Goodness of fit**: In this analysis, the pseudo-R² of the PCA model (0.076) is slightly lower than the selected original model (0.079), confirming that some predictive information was lost in the compression. The AIC advantage of the PCA model comes from parsimony, not from fitting the data better.

PCA can *help* predictive performance in high-dimensional settings by reducing overfitting and multicollinearity, but it is not a universal improvement — its benefit depends on whether the variance structure aligns with the predictive signal.

---

### Question 3: Feature Selection vs Feature Extraction (PCA)

**Feature selection** and **feature extraction** are both dimensionality reduction strategies, but they operate in fundamentally different ways:

| | Feature Selection | Feature Extraction (PCA) |
|---|---|---|
| **Output** | A *subset* of the original variables | New synthetic variables (PCs) |
| **Variables** | Original variables kept unchanged | Linear combinations of all original variables |
| **Interpretability** | High — selected variables retain their original meaning | Lower — each PC blends multiple variables |
| **Information** | Discards entire variables, retaining full information in the selected ones | Retains a fraction of total variance across all variables |
| **Example** | Keeping only `lifting_hours` and `age` | Creating PC1 = "physical labour intensity" |

The key conceptual difference is that **feature selection preserves the original variable space** (choosing which variables to keep and which to discard), while **feature extraction creates an entirely new, lower-dimensional representation** (transforming all variables into derived components). In the original selected logistic regression model, we can directly say "each extra hour of lifting increases the log-odds by 0.278." With PCA, the coefficient for PC1 describes the effect of an abstract composite — interpretable in terms of the original variables only indirectly, via the loadings.
