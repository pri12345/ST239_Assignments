# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             confusion_matrix, roc_curve, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# load data
df = pd.read_csv('Q3_OnlineShopping.csv')
print(df.shape)
print(df.dtypes)


# ==============================================================
# EDA
# ==============================================================

# quick look at the target
purchase_counts = df['purchase'].value_counts()
print("\nClass balance:\n", purchase_counts)
print(f"  -> {purchase_counts[1]/len(df)*100:.1f}% purchases, {purchase_counts[0]/len(df)*100:.1f}% no-purchase")

# class balance bar chart
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(['No Purchase (0)', 'Purchase (1)'], purchase_counts.values,
       color=['steelblue', 'darkorange'], edgecolor='white')
ax.set_ylabel('Count')
ax.set_title('Target Class Distribution')
for i, v in enumerate(purchase_counts.values):
    ax.text(i, v + 20, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# distributions of numeric features
num_cols = ['session_duration_min', 'page_views', 'items_to_cart',
            'days_since_last_visit', 'product_rating_mean']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='white')
    axes[i].set_title(col)
    axes[i].set_xlabel('')
axes[-1].set_visible(False)   # blank the 6th slot
plt.suptitle('Numeric Feature Distributions', fontsize=14)
plt.tight_layout()
plt.show()

# items_to_cart and page_views are quite right-skewed, session duration too

# now look at how the numeric features differ by purchase outcome
fig, axes = plt.subplots(1, len(num_cols), figsize=(16, 5))
for ax, col in zip(axes, num_cols):
    df.boxplot(column=col, by='purchase', ax=ax, grid=False)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel('purchase')
plt.suptitle('Numeric Features by Purchase Outcome', fontsize=13)
plt.tight_layout()
plt.show()

# categorical feature breakdown — counts split by purchase
cat_cols = ['device', 'visitor_type', 'time_of_day', 'day_type',
            'main_product_category', 'delivery_speed', 'return_policy']

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = df.groupby([col, 'purchase']).size().unstack(fill_value=0)
    ct.plot(kind='bar', ax=axes[i], color=['steelblue','darkorange'],
            edgecolor='white', width=0.7)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].legend(['No Purchase', 'Purchase'], fontsize=7)
axes[-1].set_visible(False)
plt.suptitle('Categorical Features vs Purchase', fontsize=13)
plt.tight_layout()
plt.show()

# pairplot of numeric features coloured by purchase
# (using a sample to avoid slowness)
sample = df.sample(600, random_state=42)
sns.pairplot(sample[num_cols + ['purchase']].astype({'purchase':'category'}),
             hue='purchase', diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15})
plt.suptitle('Pairplot of Numeric Features (sample n=600)', y=1.01, fontsize=12)
plt.show()


# ==============================================================
# Preprocessing
# ==============================================================

# one-hot encode all categoricals — drop_first to avoid multicollinearity
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_enc.drop(columns=['purchase'])
y = df_enc['purchase']

print("\nFeature matrix shape:", X.shape)
print("Columns:", list(X.columns))

# scale numeric columns only — tree methods don't need it but LR does
# easiest to just scale everything; doesn't hurt RF
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


# ==============================================================
# Cross-validation setup
# ==============================================================

# stratified 5-fold so class ratio is preserved in every fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_report(name, model, X, y, cv):
    """run CV and print a tidy summary — accuracy, F1, AUC"""
    scoring = ['accuracy', 'f1', 'roc_auc']
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    print(f"\n--- {name} ---")
    for s in scoring:
        vals = results[f'test_{s}']
        print(f"  {s:12s}: {vals.mean():.4f}  (+/- {vals.std():.4f})")
    return results


# ==============================================================
# Logistic Regression
# ==============================================================

# LR is a natural baseline for binary classification —
# it models log-odds as a linear combination of features
# C=1 is a reasonable default (inverse regularisation strength)
lr = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='lbfgs')

lr_results = cv_report('Logistic Regression', lr, X_scaled, y, cv)

# fit on full data for coefficient inspection
lr.fit(X_scaled, y)

# coefficients — sort by absolute magnitude to see what's driving predictions
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coef': lr.coef_[0]
}).sort_values('coef', key=abs, ascending=False)

print("\nTop 10 LR coefficients (by magnitude):")
print(coef_df.head(10).to_string(index=False))

# plot top 15
top15 = coef_df.head(15)
colors = ['darkorange' if c > 0 else 'steelblue' for c in top15['coef']]

plt.figure(figsize=(9, 6))
plt.barh(top15['feature'][::-1], top15['coef'][::-1], color=colors[::-1], edgecolor='white')
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression — Top 15 Feature Coefficients')
plt.tight_layout()
plt.show()


# ==============================================================
# Random Forest
# ==============================================================

# RF builds many decorrelated trees via bootstrap sampling + random feature subsets
# it handles non-linearities and interactions that LR would miss
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             min_samples_leaf=5, random_state=42, n_jobs=-1)

rf_results = cv_report('Random Forest', rf, X_scaled, y, cv)

# fit on full data for feature importance
rf.fit(X_scaled, y)

imp_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 RF feature importances:")
print(imp_df.head(10).to_string(index=False))

# plot top 15 importances
top15_imp = imp_df.head(15)

plt.figure(figsize=(9, 6))
plt.barh(top15_imp['feature'][::-1], top15_imp['importance'][::-1],
         color='steelblue', edgecolor='white')
plt.xlabel('Mean Decrease in Impurity')
plt.title('Random Forest — Top 15 Feature Importances')
plt.tight_layout()
plt.show()


# ==============================================================
# ROC curves + comparison
# ==============================================================

# get predicted probabilities from each fold manually so we can plot ROC
# across folds — gives a sense of variability too

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, model, name, color in zip(
        axes,
        [LogisticRegression(C=1, max_iter=1000, random_state=42),
         RandomForestClassifier(n_estimators=200, max_depth=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)],
        ['Logistic Regression', 'Random Forest'],
        ['darkorange', 'steelblue']):

    fold_aucs = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        Xtr, Xte = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(Xtr, ytr)
        probs = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, probs)
        auc = roc_auc_score(yte, probs)
        fold_aucs.append(auc)
        ax.plot(fpr, tpr, color=color, alpha=0.3, linewidth=1)

    ax.plot([0,1],[0,1],'k--', linewidth=0.8)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'{name}\nMean AUC = {np.mean(fold_aucs):.3f}')

plt.suptitle('ROC Curves across CV Folds', fontsize=13)
plt.tight_layout()
plt.show()


# ==============================================================
# Summary comparison table
# ==============================================================

metrics = ['accuracy', 'f1', 'roc_auc']
summary = {}
for name, res in [('Logistic Regression', lr_results), ('Random Forest', rf_results)]:
    summary[name] = {m: f"{res[f'test_{m}'].mean():.4f} (+/- {res[f'test_{m}'].std():.4f})"
                     for m in metrics}

summary_df = pd.DataFrame(summary).T
print("\n=== CV Performance Summary ===")
print(summary_df.to_string())

# visual comparison — mean scores bar chart
means_lr = [lr_results[f'test_{m}'].mean() for m in metrics]
means_rf = [rf_results[f'test_{m}'].mean() for m in metrics]

x = np.arange(len(metrics))
w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, means_lr, w, label='Logistic Regression', color='darkorange', edgecolor='white')
ax.bar(x + w/2, means_rf, w, label='Random Forest', color='steelblue', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'F1', 'AUC-ROC'])
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — CV Mean Scores')
ax.legend()
plt.tight_layout()
plt.show()


# confusion matrix on full data fit (just for illustration, not CV)
lr.fit(X_scaled, y)
rf.fit(X_scaled, y)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, model, name in zip(axes,
                            [lr, rf],
                            ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y, model.predict(X_scaled))
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Purchase', 'Purchase'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name} — Confusion Matrix\n(full data, for illustration)')

plt.tight_layout()
plt.show()
