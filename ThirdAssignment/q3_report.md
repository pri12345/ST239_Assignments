# Question 3: Analysis of Online Shopping Dataset

---

## 1. Introduction

The dataset contains 4,000 anonymised online shopping sessions, each representing a unique user visit recorded over a one-year period. For every session we observe a mix of numerical and categorical variables that describe the user's behaviour, device, timing, product context, and the commercial environment they browsed in. The target variable, `purchase`, is binary — it takes the value 1 if a purchase was made during the session and 0 otherwise.

The numerical features are: `session_duration_min` (how long the session lasted), `page_views` (number of pages visited), `items_to_cart` (items added to the cart), `days_since_last_visit` (recency of the previous visit), and `product_rating_mean` (average rating of viewed products on a 1–5 scale). The categorical features cover device type (mobile, desktop, tablet), visitor type (new vs. returning), time of day (morning, afternoon, evening, night), day type (weekday vs. weekend), main product category viewed (fashion, electronics, home, sports, books, beauty), delivery speed offered (standard, express, next-day), and the return policy available (basic vs. extended).

The goal of the analysis is to understand which session characteristics are associated with a purchase and to build predictive models that can distinguish purchasing from non-purchasing sessions.

---

## 2. Exploratory Data Analysis

### 2.1 Class Balance

The dataset is moderately imbalanced: 2,909 sessions (72.7%) ended without a purchase and 1,091 (27.3%) resulted in one. This imbalance is realistic for e-commerce — conversion rates are typically low — but it does mean that a naive model that always predicts "no purchase" would already achieve ~73% accuracy. This motivates looking beyond accuracy when evaluating models (see Section 3.4).

*(Figure: Target Class Distribution bar chart — produced in the `# class balance bar chart` block)*

### 2.2 Numerical Features

Summary statistics for the five numerical variables are shown below:

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| session_duration_min | 4.49 | 2.21 | 0.40 | 4.42 | 12.99 |
| page_views | 6.41 | 2.59 | 1 | 6 | 16 |
| items_to_cart | 0.50 | 0.72 | 0 | 0 | 5 |
| days_since_last_visit | 131 | 117 | 1 | 103 | 365 |
| product_rating_mean | 4.15 | 0.33 | 2.80 | 4.17 | 5.00 |

`items_to_cart` is heavily right-skewed — the median is 0 meaning most sessions involve no cart activity at all, which makes intuitive sense (browsing without buying is the norm). `days_since_last_visit` also spans a wide range (1–365 days), suggesting a mix of very frequent and infrequent shoppers in the dataset.

Looking at how these variables differ by purchase outcome (box plots, `# now look at how the numeric features differ by purchase outcome`), sessions that result in a purchase tend to be longer, involve more page views, and critically — almost always include at least one item added to the cart. The `days_since_last_visit` pattern is less clear-cut, with purchasing sessions showing slightly lower recency (meaning the user visited more recently), but with substantial overlap.

*(Figure: Numeric Features by Purchase Outcome — boxplot block)*

### 2.3 Categorical Features

Mobile is the dominant device (56% of sessions) followed by desktop (33%) and tablet (10%). New visitors outnumber returning ones (~63% vs ~37%). Most sessions occur on weekdays (~72%) and in the evening or afternoon. Fashion is the most browsed product category (~24%), followed by electronics and home goods. The majority of sessions involve standard delivery and a basic return policy.

Looking at purchase rates across categories (stacked bar charts, `# categorical feature breakdown`), returning visitors appear to have a somewhat higher conversion rate than new visitors. This is consistent with purchase intent — someone who has been to the store before may be further along in the decision process. The effect of device type is less pronounced, though mobile sessions slightly underperform desktop in terms of conversions.

*(Figure: Categorical Features vs Purchase — bar chart grid)*

### 2.4 Aims

Based on the exploratory analysis, the key signals for predicting a purchase appear to be: cart activity (`items_to_cart`), session engagement (`session_duration_min`, `page_views`), visitor recency (`days_since_last_visit`), and visitor type. The goal of the technical analysis is to quantify these associations and evaluate how well two different modelling approaches — Logistic Regression and Random Forest — can predict purchase outcomes using all available features.

---

## 3. Technical Analysis

### 3.1 Preprocessing

Since all categorical variables are nominal (no ordinal structure), they were one-hot encoded using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity. This expanded the feature matrix from 12 columns to 20 after dropping one reference category per variable.

All features were then standardised using `StandardScaler` (zero mean, unit variance). While tree-based methods like Random Forest are invariant to monotone transformations and don't strictly require scaling, standardising ensures that the Logistic Regression coefficients are directly comparable across features. Applying it to both models also keeps the pipeline consistent.

No missing values were present in the dataset, so no imputation was required.

### 3.2 Logistic Regression

Logistic Regression models the probability of a purchase using a linear combination of features passed through the logistic function:

$$P(\text{purchase} = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol{\beta}^T \mathbf{x})}}$$

The log-odds of purchasing is therefore a linear function of the predictors, making the model highly interpretable — the sign and magnitude of each coefficient directly quantifies the direction and strength of each feature's association with the outcome. We used `C=1` (the default inverse regularisation strength), which applies moderate L2 regularisation to prevent overfitting.

**Top coefficients (by magnitude):**

| Feature | Coefficient |
|---|---|
| items_to_cart | +0.859 |
| session_duration_min | +0.557 |
| visitor_type_returning | +0.376 |
| days_since_last_visit | −0.363 |
| page_views | +0.356 |
| product_rating_mean | +0.257 |
| return_policy_extended | +0.217 |
| device_mobile | −0.191 |

The largest positive coefficient belongs to `items_to_cart`, which is unsurprising — adding items to a cart is a strong signal of purchase intent. Session duration and page views both have positive effects, consistent with the idea that more engaged browsing leads to buying. Returning visitors are more likely to purchase than new ones (positive coefficient on `visitor_type_returning`). Interestingly, `days_since_last_visit` has a negative coefficient: the longer since the last visit, the *less* likely a purchase, suggesting that recency matters. Mobile device usage is negatively associated with purchase, which may reflect mobile users browsing more casually.

*(Figure: Top 15 LR Coefficients — horizontal bar chart)*

### 3.3 Random Forest

Random Forest is an ensemble of decision trees, each trained on a bootstrap sample of the data and a random subset of features at each split. The predictions are averaged across all trees, which reduces variance compared to a single tree and makes the model robust to overfitting. Unlike Logistic Regression, Random Forest can capture non-linear relationships and feature interactions without explicit specification.

We used 200 trees, `max_depth=10`, and `min_samples_leaf=5` — the depth and leaf size constraints act as regularisation to prevent individual trees from memorising the training data.

The model produces feature importance scores via the mean decrease in Gini impurity across all trees and splits. The top features by importance largely mirror the LR findings:

| Feature | Importance |
|---|---|
| items_to_cart | 0.210 |
| days_since_last_visit | 0.181 |
| session_duration_min | 0.173 |
| visitor_type_returning | 0.114 |
| page_views | 0.093 |
| product_rating_mean | 0.079 |

The top-6 features are the same across both models (modulo ranking differences), which provides good convergent evidence that these variables genuinely drive purchase behaviour. The RF places relatively more weight on `days_since_last_visit` than LR does, which may reflect non-linear recency effects that the linear model cannot capture.

*(Figure: Top 15 RF Feature Importances — horizontal bar chart)*

### 3.4 Cross-Validation and Performance

Both models were evaluated using stratified 5-fold cross-validation (to preserve the ~27% class balance in every fold). We report three metrics: accuracy, F1 score (harmonic mean of precision and recall, appropriate for imbalanced classes), and AUC-ROC (which measures discrimination ability across all classification thresholds and is invariant to class imbalance).

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.8047 ± 0.007 | 0.5911 ± 0.017 | 0.8423 ± 0.007 |
| Random Forest | 0.8053 ± 0.010 | 0.5648 ± 0.030 | 0.8314 ± 0.008 |

The two models achieve nearly identical accuracy (~80.5%), but LR outperforms RF on both F1 (+2.6 percentage points) and AUC-ROC (+1.1 pp). This is a somewhat counter-intuitive result — Random Forest is generally expected to outperform Logistic Regression on complex tabular data. A few factors may explain this:

- The dataset's decision boundary may be largely linear. With only ~20 features after encoding, there is limited room for complex interactions, and LR's smooth probabilistic output may generalise better in this regime.
- The Random Forest's lower and more variable F1 score (std 0.030 vs 0.017 for LR) suggests it is somewhat less stable, possibly overfitting individual folds slightly despite regularisation. Tuning `n_estimators` and `max_depth` further might close the gap.
- LR's AUC advantage suggests it produces better-calibrated class probability estimates, which is what AUC measures.

The ROC curves (plotted across all five folds, `# ROC curves + comparison`) show that LR curves are slightly higher and tighter across folds, consistent with the mean AUC gap.

*(Figure: ROC Curves across CV Folds — side-by-side plot)*
*(Figure: Model Comparison — CV Mean Scores bar chart)*

---

## 4. Conclusions

### Non-Technical Summary

We built two predictive models to answer the question: *can we predict whether an online shopping session will result in a purchase?* Both models agree on the key drivers: the most telling signal is whether the user added anything to their cart. Other important factors include how long they browsed, how many pages they visited, and how recently they had last visited the site. Returning visitors are also more likely to buy than first-time visitors. Device type plays a smaller role, with mobile users slightly less likely to convert than desktop users.

Overall, the models correctly identify purchasing sessions about 80% of the time. The Logistic Regression model performs marginally better than the Random Forest in this setting, achieving a higher F1 and AUC score. Both models, however, still miss a meaningful fraction of purchases (low recall on the positive class), which is partly a consequence of the imbalanced class distribution.

### Expert Discussion

The convergence of feature rankings across LR and RF is reassuring — it suggests the relationships are genuinely informative rather than artefacts of one model's assumptions. The dominance of `items_to_cart` and engagement metrics is consistent with established e-commerce research on purchase intent.

**Limitations:**

- *Class imbalance*: with only ~27% positive cases, F1 scores in the 0.56–0.59 range reflect the challenge of detecting the minority class. Techniques like SMOTE (oversampling), class-weighted loss functions (`class_weight='balanced'`), or threshold adjustment could improve recall on the positive class.
- *No hyperparameter tuning for LR or RF*: the current RF uses default-ish settings; a proper grid search over `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features` would likely improve performance. LR could also benefit from tuning `C` via cross-validated regularisation path.
- *Feature engineering*: interaction terms (e.g. `items_to_cart × session_duration`) or polynomial features could expose non-linearities to LR, potentially narrowing the gap with tree-based approaches.

**Potential extensions:**

- Gradient Boosting methods (XGBoost, LightGBM) typically outperform standard RF on tabular data and would be a natural next step.
- Calibrating the RF's probability outputs (via Platt scaling or isotonic regression) could improve its AUC score, which currently lags LR's.
- Given that `days_since_last_visit` is an important feature, survival analysis or a time-aware model might provide richer insight into purchase timing patterns.
