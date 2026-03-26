# Question 3: Analysis of Online Shopping Dataset

---

## 1. Introduction

The dataset contains 4,000 anonymised online shopping sessions, each representing a unique user visit recorded over a one-year period. For every session we observe a combination of numerical and categorical variables describing the user's behaviour, the device they used, the timing of the visit, and the product context they were browsing in. The target variable, `purchase`, is binary: 1 if the session ended in a purchase, 0 otherwise.

The five numerical features are `session_duration_min` (length of the session), `page_views` (pages visited), `items_to_cart` (items added to the shopping cart), `days_since_last_visit` (time since the previous session), and `product_rating_mean` (average rating of viewed products on a 1-5 scale). On the categorical side, we have device type (mobile, desktop, tablet), visitor type (new vs. returning), time of day (morning, afternoon, evening, night), day type (weekday vs. weekend), main product category (fashion, electronics, home, sports, books, beauty), delivery speed (standard, express, next-day), and return policy (basic vs. extended).

The goal of the analysis is to understand which session characteristics are most associated with a purchase and to evaluate how well two different modelling approaches can predict purchase outcomes from these features.

---

## 2. Exploratory Data Analysis

### 2.1 Class Balance

Of the 4,000 sessions, 2,909 (72.7%) did not result in a purchase and 1,091 (27.3%) did. This kind of imbalance is typical in e-commerce, where most browsing sessions never convert to a sale. It does create a problem for model evaluation though: a model that simply predicts "no purchase" every time would already be right 73% of the time, so accuracy alone is not a meaningful performance metric here. We need to look at F1 and AUC-ROC as well (see Section 3.4).

[placeholder for image: target class distribution bar chart]

### 2.2 Numerical Features

Summary statistics for the five numerical variables are shown below:

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| session_duration_min | 4.49 | 2.21 | 0.40 | 4.42 | 12.99 |
| page_views | 6.41 | 2.59 | 1 | 6 | 16 |
| items_to_cart | 0.50 | 0.72 | 0 | 0 | 5 |
| days_since_last_visit | 131 | 117 | 1 | 103 | 365 |
| product_rating_mean | 4.15 | 0.33 | 2.80 | 4.17 | 5.00 |

`items_to_cart` stands out immediately: the median is 0, meaning more than half of all sessions involve no cart activity whatsoever. The distribution is heavily right-skewed, with a small number of sessions accounting for most of the cart additions. `days_since_last_visit` covers the full range from 1 to 365 days, pointing to a mix of habitual shoppers and people who haven't visited in a long time.

Looking at how these features differ by purchase outcome (boxplots, `# boxplots by purchase`), sessions that end in a purchase tend to be longer, involve more page views, and almost always include at least one cart addition. The separation on `items_to_cart` is particularly sharp. The `days_since_last_visit` pattern is less clean-cut: purchasing sessions skew slightly toward more recent visits, but there is plenty of overlap.

[placeholder for image: numeric features by purchase outcome, boxplots]

### 2.3 Categorical Features

Mobile is the most common device (roughly 56% of sessions), followed by desktop (33%) and tablet (10%). New visitors outnumber returning ones at around 63% vs 37%, and the majority of sessions occur on weekdays and in the evening or afternoon. Fashion is the most frequently browsed product category, followed by electronics and home goods.

When purchase rates are broken down by category (bar charts, `# cats vs purchase`), returning visitors appear to convert at a noticeably higher rate than new ones, which makes sense: someone visiting for a second or third time is probably further along in their decision-making. The device effect is more modest, though mobile sessions do slightly underperform desktop in terms of conversions, possibly because mobile browsing tends to be more casual and exploratory.

[placeholder for image: categorical features vs purchase, bar chart grid]

### 2.4 Aims

The exploratory analysis points to a few features as likely strong predictors: cart activity (`items_to_cart`), session engagement (`session_duration_min`, `page_views`), recency (`days_since_last_visit`), and visitor type. The technical analysis that follows aims to quantify these associations and compare two modelling approaches, Logistic Regression and Random Forest, in terms of their ability to predict purchases from the full set of features.

---

## 3. Technical Analysis

### 3.1 Preprocessing

All categorical variables in this dataset are nominal (there is no natural ordering to device type or product category), so they were one-hot encoded using `pd.get_dummies` with `drop_first=True`. Dropping the first dummy per variable avoids the dummy variable trap, where one category is perfectly predictable from the others. After encoding, the feature matrix expands from 12 columns to 20.

All features were then standardised with `StandardScaler` (zero mean, unit variance). Strictly speaking, tree-based methods like Random Forest are scale-invariant and do not need this step. Applying it to both models anyway keeps the pipeline consistent and ensures that Logistic Regression coefficients are on a comparable scale, which makes interpreting their magnitudes more straightforward. No missing values were found in the dataset, so no imputation was needed.

### 3.2 Logistic Regression

Logistic Regression models the probability of a purchase as:

$$P(\text{purchase} = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol{\beta}^T \mathbf{x})}}$$

The log-odds of a purchase is a linear function of the features, which makes the model directly interpretable: positive coefficients increase the probability of purchasing, negative ones decrease it, and the magnitude reflects how strongly each feature matters. We used `C=1` (the default inverse regularisation strength), applying moderate L2 regularisation to discourage large coefficients and reduce overfitting.

The top coefficients by absolute magnitude are:

| Feature | Coefficient |
|---|---|
| items_to_cart | +0.859 |
| session_duration_min | +0.557 |
| visitor_type_returning | +0.376 |
| days_since_last_visit | -0.363 |
| page_views | +0.356 |
| product_rating_mean | +0.257 |
| return_policy_extended | +0.217 |
| device_mobile | -0.191 |

`items_to_cart` has by far the largest coefficient, which is intuitive: adding something to a cart is a very direct signal of purchase intent. Session duration and page views both contribute positively, consistent with the idea that more engaged browsing leads to buying. Returning visitors are more likely to purchase than new ones. `days_since_last_visit` carries a negative coefficient: the longer since the last visit, the less likely a purchase, suggesting recency matters. Mobile users are slightly less likely to convert than desktop users, consistent with the EDA finding above.

[placeholder for image: top 15 LR coefficients, horizontal bar chart]

### 3.3 Random Forest

Random Forest builds a large collection of decision trees, each trained on a bootstrap sample of the data and using a random subset of features at each split. The final prediction is an average over all trees, which substantially reduces variance compared to any single tree. The key advantage over Logistic Regression is the ability to capture non-linear relationships and feature interactions without needing to specify them explicitly.

We used 200 trees with `max_depth=10` and `min_samples_leaf=5`. The depth and leaf-size constraints act as regularisation, preventing individual trees from growing too complex and memorising the training data.

Feature importance in Random Forest is measured by the mean decrease in Gini impurity across all trees and splits. The top features by importance are:

| Feature | Importance |
|---|---|
| items_to_cart | 0.210 |
| days_since_last_visit | 0.181 |
| session_duration_min | 0.173 |
| visitor_type_returning | 0.114 |
| page_views | 0.093 |
| product_rating_mean | 0.079 |

The same six features dominate in both models. One notable difference in ranking is that `days_since_last_visit` is placed second by the Random Forest but fourth by Logistic Regression, which likely reflects non-linear recency effects that the tree model can capture but the linear model cannot.

[placeholder for image: top 15 RF feature importances, horizontal bar chart]

### 3.4 Cross-Validation and Performance

Both models were evaluated using stratified 5-fold cross-validation, which ensures that the ~27% purchase rate is preserved in every fold. We report accuracy, F1 score (harmonic mean of precision and recall, which handles class imbalance more fairly than accuracy), and AUC-ROC (which measures discrimination ability across all classification thresholds and is unaffected by the class imbalance).

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.8047 ± 0.0066 | 0.5911 ± 0.0166 | 0.8423 ± 0.0068 |
| Random Forest | 0.8053 ± 0.0095 | 0.5648 ± 0.0298 | 0.8314 ± 0.0078 |

The two models achieve almost identical accuracy (~80.5%), but Logistic Regression comes out ahead on both F1 (+2.6 percentage points) and AUC-ROC (+1.1 pp). This is a somewhat surprising result given that Random Forest is generally expected to outperform a linear model on tabular data. The reasons behind it are discussed in Section 4.4.

The ROC curves plotted across all five folds (`# ROC curves`) show the LR curves sitting slightly higher and closer together than the RF curves, consistent with the mean AUC gap.

[placeholder for image: ROC curves across CV folds, side-by-side for LR and RF]

[placeholder for image: model comparison bar chart, CV mean scores]

---

## 4. Technical Reflections

### 4.1 Underlying Assumptions

**Logistic Regression** rests on several assumptions that are worth making explicit. The central one is linearity in the log-odds: the model assumes that $\log\frac{p}{1-p}$ is a linear function of the predictors. This is not testable directly from the output, but the fact that LR performs comparably to (and in this case better than) a non-linear model is some evidence that the assumption is not badly violated here. A second assumption is independence of observations: each session is treated as contributing independent information. This is plausible given that the dataset explicitly has one session per user over the year, so we do not expect repeated-user effects to inflate the apparent sample size. Third, the model requires no perfect multicollinearity among the predictors. This is exactly why `drop_first=True` was used during one-hot encoding: including all dummy levels for a categorical variable would create a column that is a perfect linear combination of the others, making the coefficient matrix singular. Residual correlations between features (for instance, `page_views` and `session_duration_min` are likely correlated) are handled by the L2 regularisation, which spreads coefficient mass across correlated predictors rather than assigning large weights arbitrarily. Finally, with 4,000 observations and only 20 features, we are comfortably in the large-sample regime where the asymptotic properties of maximum likelihood estimates hold well.

**Random Forest** has far fewer formal assumptions. It is non-parametric: no distributional form is assumed for the features, and the model will handle skewed, bounded, or discrete variables without transformation. The main requirement is that observations are independently and identically distributed (IID), which is satisfied here for the same reasons as above. The key mechanism behind RF's effectiveness is the decorrelation of individual trees, achieved by restricting each split to a random subset of features (by default $\sqrt{p}$ features for classification). If all trees were grown on the same features, they would be highly correlated, and averaging them would not reduce variance much. The random subsets break this correlation, meaning errors in individual trees tend to cancel rather than compound. A practical assumption is also that enough signal exists in the features to split on: if all features were pure noise, the trees would be random and the ensemble would not outperform a naive baseline.

### 4.2 Variable Selection

All 20 post-encoding features were retained without explicit selection. This is justifiable for both models. For Logistic Regression, L2 regularisation shrinks the coefficients of weakly predictive features toward zero, effectively reducing their influence on predictions without dropping them outright. For Random Forest, the random feature subsets at each split mean that uninformative features are selected infrequently and contribute little to the final ensemble even if they are technically included.

The convergence of both models on the same top-6 features (items_to_cart, days_since_last_visit, session_duration_min, visitor_type_returning, page_views, product_rating_mean) provides a form of post-hoc validation of this approach: if the 20-feature set were introducing substantial noise, we would expect the two models, with their very different inductive biases, to produce inconsistent importance rankings. The agreement instead suggests the predictive signal is concentrated in a small, coherent subset of features that both methods identify robustly.

A more principled alternative would have been recursive feature elimination (RFE), using the RF importances or LR coefficients to iteratively prune the weakest features and re-evaluate CV performance at each step. Given the modest feature count and the implicit regularisation already in place, this was not pursued, but it could be a useful sensitivity check.

### 4.3 Hyperparameter Choices

**Logistic Regression** was fitted with `C=1`, which is the sklearn default. `C` is the inverse of the regularisation strength: smaller values impose stronger L2 shrinkage (more bias, less variance), larger values allow coefficients to grow more freely (less bias, more variance). A value of C=1 is a reasonable middle ground for well-scaled data, but it was not tuned. A proper approach would fit a regularisation path over a logarithmic grid (e.g. $C \in \{0.01, 0.1, 1, 10, 100\}$) and select the value that maximises cross-validated AUC. The solver `lbfgs` was used, which is appropriate for small-to-medium datasets with L2 regularisation.

**Random Forest** was fitted with `n_estimators=200`, `max_depth=10`, and `min_samples_leaf=5`. The number of trees (200) was chosen to be large enough that the ensemble error has plateaued: adding more trees beyond this point typically yields diminishing returns, and 200 is generally sufficient for datasets of this size. `max_depth=10` caps how deep individual trees can grow, which prevents them from forming very specific decision rules that only hold in the training data. `min_samples_leaf=5` requires each terminal node to contain at least 5 observations, which has a similar regularising effect and also ensures that leaf-level probability estimates are based on a reasonable number of samples. Both constraints were chosen as reasonable defaults rather than tuned values. A grid search over, say, `max_depth` in {5, 10, 15, None} and `min_samples_leaf` in {1, 5, 10} would likely refine performance, particularly the F1 score where the model shows more variability across folds.

### 4.4 Why Logistic Regression Outperforms Random Forest

The performance gap (LR leads on F1 and AUC) warrants a closer look, since the conventional expectation is that ensemble tree methods outperform linear models on real-world tabular data.

One likely explanation is that the decision boundary in this problem is approximately linear. After one-hot encoding, the feature space has 20 dimensions, and the key predictors (items_to_cart, session_duration, page_views) all show fairly monotone relationships with the purchase probability — which is exactly the structure that LR is built for. With $p=20$, the number of candidate interaction terms is $\binom{20}{2} = 190$, and RF would need enough data in each region of feature space to learn these interactions reliably. At 4,000 observations, this may be borderline.

A second factor is probability calibration. AUC-ROC measures the model's ability to correctly rank a randomly chosen positive case above a randomly chosen negative one, which is a direct function of the predicted probabilities. LR's logistic output is inherently well-calibrated: its sigmoid maps the linear score to a probability in a smooth, monotone way. Decision trees, by contrast, produce probabilities as leaf frequencies, which tend to be overconfident (pushed toward 0 and 1) because each leaf is fit to a small subset of training data. Averaging over 200 trees in RF partially corrects this, but LR's structural advantage in calibration is consistent with its higher AUC.

The third factor is fold-level stability. LR's F1 standard deviation across folds is 0.017, compared to 0.030 for RF. This higher variance in RF suggests that the ensemble is somewhat sensitive to which observations fall in each fold, possibly because the class imbalance (27% positives) means that small fluctuations in minority-class composition across folds affect the tree-growing process more than they affect the simpler LR optimisation.

### 4.5 Class Imbalance and Its Effects

The 73/27 class split affects the models at two levels. At the **evaluation level**, raw accuracy is uninformative (a constant "no purchase" predictor scores 72.7%), which is why F1 and AUC-ROC were prioritised. At the **training level**, both LR's cross-entropy loss and RF's Gini impurity are not explicitly penalised for imbalance: they minimise overall error, which tends to push the model toward the majority class. The practical consequence is that the models' F1 scores (0.59 for LR, 0.56 for RF) are held back by relatively low recall on the positive class, even where precision is reasonable.

Several strategies could address this. The simplest is setting `class_weight='balanced'` in both sklearn models, which rescales the loss so that each class contributes equally regardless of frequency. A more aggressive option is SMOTE (Synthetic Minority Over-sampling Technique), which generates synthetic positive-class samples by interpolating between existing ones in feature space, effectively augmenting the training set until the classes are balanced. A third lever is threshold adjustment: the default decision threshold of 0.5 can be shifted downward (e.g. to 0.3), so that the model labels a session as a purchase whenever the predicted probability exceeds the lower bar. This increases recall at the cost of precision, and the optimal threshold can be selected by inspecting the precision-recall curve. Each of these interventions trades off differently and would need to be evaluated on the same CV framework used here.

---

## 5. Conclusions

### Non-Technical Summary

We built two predictive models to answer the question: can we tell from session-level data whether an online shopping visit will result in a purchase? Both models agree on what matters most. The clearest signal is whether the user added anything to their cart. Beyond that, longer and more page-heavy sessions are more likely to end in a purchase, as are sessions from returning visitors and from users who visited the site recently. Device type plays a smaller role, with mobile users converting slightly less often than desktop users.

Both models correctly identify purchasing sessions around 80% of the time. The Logistic Regression performs slightly better than the Random Forest in terms of F1 and AUC, making it the marginally preferred model in this setting. That said, both models still miss a fair proportion of purchases, partly because the class imbalance makes the minority class harder to detect.

### Expert Discussion

On the whole, both methods produce consistent and interpretable results. The agreement on feature rankings across two models with very different inductive biases (linear vs. non-linear, parametric vs. non-parametric) is encouraging: it suggests the identified predictors reflect genuine structure in the data rather than quirks of any one modelling approach.

**Potential extensions:**

- Gradient Boosting methods (XGBoost, LightGBM) typically outperform standard Random Forests on tabular data through sequential error correction rather than parallel averaging, and would be a natural next step in complexity.
- Given the importance of `days_since_last_visit`, a survival analysis or time-aware model could offer richer insight into purchase timing patterns beyond what a session-level classifier captures.
- Exploring interaction features (e.g. `items_to_cart` × `session_duration_min`) could expose non-linear structure to the Logistic Regression, potentially narrowing its performance gap with tree-based methods.
