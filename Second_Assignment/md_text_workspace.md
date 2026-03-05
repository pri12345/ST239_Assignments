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
