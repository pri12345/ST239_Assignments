import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('Q3_OnlineShopping.csv')
print(df.shape)
print(df.dtypes)


# EDA

purchase_counts = df['purchase'].value_counts()
print("\nClass balance:\n", purchase_counts)
print(f"  -> {purchase_counts[1]/len(df)*100:.1f}% purchases, {purchase_counts[0]/len(df)*100:.1f}% no-purchase")

# bar chart for target
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(['No Purchase (0)', 'Purchase (1)'], purchase_counts.values,
       color=['steelblue', 'darkorange'], edgecolor='white')
ax.set_ylabel('Count')
ax.set_title('Target Class Distribution')
for i, v in enumerate(purchase_counts.values):
    ax.text(i, v + 20, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.show()

num_cols = ['session_duration_min', 'page_views', 'items_to_cart',
            'days_since_last_visit', 'product_rating_mean']

# histograms for numeric features
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='white')
    axes[i].set_title(col)
axes[-1].set_visible(False)
plt.suptitle('Numeric Feature Distributions', fontsize=14)
plt.tight_layout()
plt.show()

# boxplots by purchase - useful to spot which features separate the classes
fig, axes = plt.subplots(1, len(num_cols), figsize=(16, 5))
for ax, col in zip(axes, num_cols):
    df.boxplot(column=col, by='purchase', ax=ax, grid=False)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel('purchase')
plt.suptitle('Numeric Features by Purchase Outcome', fontsize=13)
plt.tight_layout()
plt.show()

cat_cols = ['device', 'visitor_type', 'time_of_day', 'day_type',
            'main_product_category', 'delivery_speed', 'return_policy']

# cats vs purchase - count bars split by outcome
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = df.groupby([col, 'purchase']).size().unstack(fill_value=0)
    ct.plot(kind='bar', ax=axes[i], color=['steelblue', 'darkorange'], edgecolor='white', width=0.7)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].legend(['No Purchase', 'Purchase'], fontsize=7)
axes[-1].set_visible(False)
plt.suptitle('Categorical Features vs Purchase', fontsize=13)
plt.tight_layout()
plt.show()

# pairplot on a sample - takes a bit to render
samp = df.sample(600, random_state=42)
sns.pairplot(samp[num_cols + ['purchase']].astype({'purchase': 'category'}),
             hue='purchase', diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15})
plt.suptitle('Pairplot of Numeric Features (sample n=600)', y=1.01, fontsize=12)
plt.show()


# Preprocessing

# one-hot encode categoricals, drop_first to avoid collinearity
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
X = df_enc.drop(columns=['purchase'])
y = df_enc['purchase']

print("\nFeature matrix shape:", X.shape)
print("Columns:", list(X.columns))

# scale everything - LR needs it, doesnt really hurt RF
sc = StandardScaler()
X_sc = sc.fit_transform(X)
X_sc = pd.DataFrame(X_sc, columns=X.columns)


# Cross-validation setup

# stratified so the 27/73 split is preserved across folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_report(name, model, X, y, cv):
    res = cross_validate(model, X, y, cv=cv,
                         scoring=['accuracy', 'f1', 'roc_auc'],
                         return_train_score=False)
    print(f"\n--- {name} ---")
    for s in ['accuracy', 'f1', 'roc_auc']:
        v = res[f'test_{s}']
        print(f"  {s:12s}: {v.mean():.4f}  (+/- {v.std():.4f})")
    return res


# Logistic Regression

# models log-odds as a linear combo 
lr = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='lbfgs')
lr_res = cv_report('Logistic Regression', lr, X_sc, y, cv)

# fit on full data to look at coefficients
lr.fit(X_sc, y)

coef_df = pd.DataFrame({'feature': X.columns, 'coef': lr.coef_[0]}).sort_values('coef', key=abs, ascending=False)
print("\nTop 10 LR coefficients (by magnitude):")
print(coef_df.head(10).to_string(index=False))

top15 = coef_df.head(15)
cols = ['darkorange' if c > 0 else 'steelblue' for c in top15['coef']]
plt.figure(figsize=(9, 6))
plt.barh(top15['feature'][::-1], top15['coef'][::-1], color=cols[::-1], edgecolor='white')
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression - Top 15 Feature Coefficients')
plt.tight_layout()
plt.show()


# Random Forest

# ensemble of trees, handles non-linearities and interactions LR cant
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                            min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_res = cv_report('Random Forest', rf, X_sc, y, cv)

rf.fit(X_sc, y)

imp_df = pd.DataFrame({'feature': X.columns,
                       'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
print("\nTop 10 RF feature importances:")
print(imp_df.head(10).to_string(index=False))

plt.figure(figsize=(9, 6))
plt.barh(imp_df['feature'][:15][::-1], imp_df['importance'][:15][::-1],
         color='steelblue', edgecolor='white')
plt.xlabel('Mean Decrease in Impurity')
plt.title('Random Forest - Top 15 Feature Importances')
plt.tight_layout()
plt.show()


# ROC Curves

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model, name, color in zip(
        axes,
        [LogisticRegression(C=1, max_iter=1000, random_state=42),
         RandomForestClassifier(n_estimators=200, max_depth=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)],
        ['Logistic Regression', 'Random Forest'],
        ['darkorange', 'steelblue']):

    aucs = []
    for train_idx, test_idx in cv.split(X_sc, y):
        Xtr, Xte = X_sc.iloc[train_idx], X_sc.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(Xtr, ytr)
        probs = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, probs)
        aucs.append(roc_auc_score(yte, probs))
        ax.plot(fpr, tpr, color=color, alpha=0.3, linewidth=1)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{name}\nMean AUC = {np.mean(aucs):.3f}')

plt.suptitle('ROC Curves across CV Folds', fontsize=13)
plt.tight_layout()
plt.show()


# Summary comparison

metrics = ['accuracy', 'f1', 'roc_auc']
summary = {}
for name, res in [('Logistic Regression', lr_res), ('Random Forest', rf_res)]:
    summary[name] = {m: f"{res[f'test_{m}'].mean():.4f} (+/- {res[f'test_{m}'].std():.4f})" for m in metrics}

print("\n=== CV Performance Summary ===")
print(pd.DataFrame(summary).T.to_string())

means_lr = [lr_res[f'test_{m}'].mean() for m in metrics]
means_rf = [rf_res[f'test_{m}'].mean() for m in metrics]
x = np.arange(len(metrics)); w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, means_lr, w, label='Logistic Regression', color='darkorange', edgecolor='white')
ax.bar(x + w/2, means_rf, w, label='Random Forest', color='steelblue', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(['Accuracy', 'F1', 'AUC-ROC'])
ax.set_ylim(0.5, 1.0); ax.set_ylabel('Score')
ax.set_title('Model Comparison - CV Mean Scores')
ax.legend()
plt.tight_layout()
plt.show()

# confusion matrices on full data fit (just to visualise, not CV)
lr.fit(X_sc, y); rf.fit(X_sc, y)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, model, name in zip(axes, [lr, rf], ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y, model.predict(X_sc))
    ConfusionMatrixDisplay(cm, display_labels=['No Purchase', 'Purchase']).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name} - Confusion Matrix (full data, illustration only)')
plt.tight_layout()
plt.show()
