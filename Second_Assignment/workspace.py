# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier

# Path setup
credit_path = Path("./credit_default.csv")

# Dataframe setup
credit_df = pd.read_csv(credit_path)
X = credit_df.drop(columns=['default'])
y = credit_df['default']

# ============================================================
# Q1: Bagging vs Boosting for Credit Default Prediction
# ============================================================

print(f"Dataset: {credit_df.shape[0]} observations, {credit_df.shape[1] - 1} features")
print(f"Class balance -- Default: {y.mean():.2%} | No Default: {1 - y.mean():.2%}")
print(credit_df.head(3))

# ============================================================
# 1.1: Hyperparameter Tuning -- Random Forest (5-Fold CV)
# ============================================================

rf_param_grid = {
    'n_estimators':     [50, 200, 500, 1000],
    'max_depth':        [2, 3, 5, None],         # None = grow until leaves are pure
    'max_features':     [3, 5, 9],               # ~sqrt(13), ~40%, ~70% of 13 features
    'min_samples_leaf': [1, 5, 10]
}

rf_base = RandomForestClassifier(criterion='log_loss', random_state=42, n_jobs=-1)

rf_grid = GridSearchCV(
    rf_base, rf_param_grid,
    cv=5,
    scoring=['neg_log_loss', 'accuracy', 'roc_auc'],
    refit='neg_log_loss',
    return_train_score=True,
    n_jobs=-1, verbose=1
)
rf_grid.fit(X, y)

rf_results  = pd.DataFrame(rf_grid.cv_results_)
best_rf_idx = rf_grid.best_index_

print(f"\n{'='*60}")
print("RANDOM FOREST -- Best Parameters (log-loss criterion):")
for k, v in rf_grid.best_params_.items():
    print(f"   {k}: {v}")
print(f"   CV Log-Loss: {-rf_grid.best_score_:.4f}")
print(f"   CV Accuracy: {rf_results.loc[best_rf_idx, 'mean_test_accuracy']:.4f}")
print(f"   CV AUC:      {rf_results.loc[best_rf_idx, 'mean_test_roc_auc']:.4f}")

# Top 5 by Accuracy and by AUC
_rf_cols_raw  = ['param_n_estimators', 'param_max_depth', 'param_max_features',
                 'param_min_samples_leaf', 'mean_test_accuracy', 'mean_test_roc_auc',
                 'mean_test_neg_log_loss']
_rf_cols_nice = ['n_est', 'max_depth', 'max_feat', 'min_leaf', 'CV Acc', 'CV AUC', 'CV NegLL']

rf_top_acc = rf_results.nlargest(5, 'mean_test_accuracy')[_rf_cols_raw].reset_index(drop=True)
rf_top_acc.columns = _rf_cols_nice
rf_top_auc = rf_results.nlargest(5, 'mean_test_roc_auc')[_rf_cols_raw].reset_index(drop=True)
rf_top_auc.columns = _rf_cols_nice

print(f"\nTop 5 RF by Accuracy:\n{rf_top_acc.round(4).to_string(index=False)}")
print(f"\nTop 5 RF by AUC:\n{rf_top_auc.round(4).to_string(index=False)}")

# ============================================================
# 1.2: Hyperparameter Tuning -- XGBoost (5-Fold CV)
# ============================================================

xgb_param_grid = {
    'n_estimators':  [50, 200, 500, 1000],
    'max_depth':     [2, 3, 4, 10],       # 10 approximates None (XGBoost requires positive int)
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample':     [0.7, 1.0]
}

xgb_base = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)

xgb_grid = GridSearchCV(
    xgb_base, xgb_param_grid,
    cv=5,
    scoring=['neg_log_loss', 'accuracy', 'roc_auc'],
    refit='neg_log_loss',
    return_train_score=True,
    n_jobs=-1, verbose=1
)
xgb_grid.fit(X, y)

xgb_results  = pd.DataFrame(xgb_grid.cv_results_)
best_xgb_idx = xgb_grid.best_index_

print(f"\n{'='*60}")
print("XGBOOST -- Best Parameters (log-loss criterion):")
for k, v in xgb_grid.best_params_.items():
    print(f"   {k}: {v}")
print(f"   CV Log-Loss: {-xgb_grid.best_score_:.4f}")
print(f"   CV Accuracy: {xgb_results.loc[best_xgb_idx, 'mean_test_accuracy']:.4f}")
print(f"   CV AUC:      {xgb_results.loc[best_xgb_idx, 'mean_test_roc_auc']:.4f}")

_xgb_cols_raw  = ['param_n_estimators', 'param_max_depth', 'param_learning_rate',
                  'param_subsample', 'mean_test_accuracy', 'mean_test_roc_auc',
                  'mean_test_neg_log_loss']
_xgb_cols_nice = ['n_est', 'max_depth', 'lr', 'subsample', 'CV Acc', 'CV AUC', 'CV NegLL']

xgb_top_acc = xgb_results.nlargest(5, 'mean_test_accuracy')[_xgb_cols_raw].reset_index(drop=True)
xgb_top_acc.columns = _xgb_cols_nice
xgb_top_auc = xgb_results.nlargest(5, 'mean_test_roc_auc')[_xgb_cols_raw].reset_index(drop=True)
xgb_top_auc.columns = _xgb_cols_nice

print(f"\nTop 5 XGB by Accuracy:\n{xgb_top_acc.round(4).to_string(index=False)}")
print(f"\nTop 5 XGB by AUC:\n{xgb_top_auc.round(4).to_string(index=False)}")

# ============================================================
# 1.3: RF Analysis -- Accuracy & AUC vs n_estimators by max_depth
# ============================================================

# Convert None to string so groupby/labels work cleanly
rf_plot = rf_results.copy()
rf_plot['param_max_depth'] = rf_plot['param_max_depth'].astype(str)

rf_by_depth = rf_plot.groupby(
    ['param_max_depth', 'param_n_estimators'], dropna=False
).agg(
    cv_acc   =('mean_test_accuracy',  'mean'),
    cv_auc   =('mean_test_roc_auc',   'mean'),
    train_acc=('mean_train_accuracy', 'mean')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Random Forest: CV Accuracy & AUC vs n_estimators\n(averaged over max_features, min_samples_leaf)', fontsize=12)

for depth, grp in rf_by_depth.groupby('param_max_depth'):
    grp = grp.sort_values('param_n_estimators')
    axes[0].plot(grp['param_n_estimators'], grp['cv_acc'], marker='o', label=f'max_depth={depth}')
    axes[1].plot(grp['param_n_estimators'], grp['cv_auc'], marker='o', label=f'max_depth={depth}')

for ax, ylabel, title in zip(
    axes,
    ['CV Accuracy', 'CV ROC-AUC'],
    ['Accuracy vs n_estimators', 'AUC vs n_estimators']
):
    ax.set_xlabel('n_estimators')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# RF: effect of min_samples_leaf (averaged over all other params)
rf_by_leaf = rf_plot.groupby('param_min_samples_leaf').agg(
    cv_acc   =('mean_test_accuracy',  'mean'),
    cv_auc   =('mean_test_roc_auc',   'mean'),
    train_acc=('mean_train_accuracy', 'mean')
).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rf_by_leaf['param_min_samples_leaf'], rf_by_leaf['cv_acc'],    marker='o', label='CV Accuracy')
ax.plot(rf_by_leaf['param_min_samples_leaf'], rf_by_leaf['cv_auc'],    marker='s', label='CV AUC')
ax.plot(rf_by_leaf['param_min_samples_leaf'], rf_by_leaf['train_acc'], marker='^', linestyle='--', label='Train Accuracy')
ax.set_xlabel('min_samples_leaf')
ax.set_ylabel('Score')
ax.set_title('RF: Accuracy, AUC and Train Score vs min_samples_leaf')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 1.4: XGBoost Analysis -- Accuracy & AUC vs n_estimators by learning_rate
# ============================================================

xgb_by_lr = xgb_results.groupby(
    ['param_learning_rate', 'param_n_estimators']
).agg(
    cv_acc   =('mean_test_accuracy',  'mean'),
    cv_auc   =('mean_test_roc_auc',   'mean'),
    train_acc=('mean_train_accuracy', 'mean')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('XGBoost: CV Accuracy & AUC vs n_estimators\n(averaged over max_depth, subsample)', fontsize=12)

for lr, grp in xgb_by_lr.groupby('param_learning_rate'):
    grp = grp.sort_values('param_n_estimators')
    axes[0].plot(grp['param_n_estimators'], grp['cv_acc'], marker='o', label=f'lr={lr}')
    axes[1].plot(grp['param_n_estimators'], grp['cv_auc'], marker='o', label=f'lr={lr}')

for ax, ylabel, title in zip(
    axes,
    ['CV Accuracy', 'CV ROC-AUC'],
    ['Accuracy vs n_estimators', 'AUC vs n_estimators']
):
    ax.set_xlabel('n_estimators')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 1.5: Overfitting Check -- Train vs CV Accuracy by max_depth
# ============================================================

rf_overfit = rf_plot.groupby('param_max_depth', dropna=False).agg(
    train_acc=('mean_train_accuracy', 'mean'),
    cv_acc   =('mean_test_accuracy',  'mean'),
).reset_index()
rf_overfit['gap'] = rf_overfit['train_acc'] - rf_overfit['cv_acc']

xgb_overfit = xgb_results.groupby('param_max_depth').agg(
    train_acc=('mean_train_accuracy', 'mean'),
    cv_acc   =('mean_test_accuracy',  'mean'),
).reset_index()
xgb_overfit['gap'] = xgb_overfit['train_acc'] - xgb_overfit['cv_acc']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Overfitting Check: Train vs CV Accuracy by max_depth', fontsize=13)

for ax, df, title in zip(axes, [rf_overfit, xgb_overfit], ['Random Forest', 'XGBoost']):
    x = np.arange(len(df))
    ax.bar(x - 0.2, df['train_acc'], 0.4, label='Train Accuracy', color='steelblue')
    ax.bar(x + 0.2, df['cv_acc'],    0.4, label='CV Accuracy',    color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(df['param_max_depth'].astype(str))
    ax.set_xlabel('max_depth')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.5, 1.05)

plt.tight_layout()
plt.show()

print("RF -- Train vs CV Accuracy gap by max_depth:")
print(rf_overfit[['param_max_depth', 'train_acc', 'cv_acc', 'gap']].round(4).to_string(index=False))
print("\nXGB -- Train vs CV Accuracy gap by max_depth:")
print(xgb_overfit[['param_max_depth', 'train_acc', 'cv_acc', 'gap']].round(4).to_string(index=False))

# ============================================================
# 1.6: Model Comparison -- Top 5 Feature Importances
# ============================================================

feature_names = X.columns.tolist()

rf_imp  = pd.Series(rf_grid.best_estimator_.feature_importances_,  index=feature_names).sort_values(ascending=False)
xgb_imp = pd.Series(xgb_grid.best_estimator_.feature_importances_, index=feature_names).sort_values(ascending=False)

print(f"\nTop 5 Features -- Random Forest (best by log-loss):\n{rf_imp.head(5).round(4).to_string()}")
print(f"\nTop 5 Features -- XGBoost (best by log-loss):\n{xgb_imp.head(5).round(4).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Top 5 Feature Importances: RF vs XGBoost (best model by log-loss)', fontsize=13)

sns.barplot(x=rf_imp.head(5).values,  y=rf_imp.head(5).index,  ax=axes[0], palette='Blues_r')
axes[0].set_title('Random Forest')
axes[0].set_xlabel('Importance')

sns.barplot(x=xgb_imp.head(5).values, y=xgb_imp.head(5).index, ax=axes[1], palette='Oranges_r')
axes[1].set_title('XGBoost')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()

# Full ranking side-by-side
rank_df = pd.DataFrame({
    'RF Importance':  rf_imp.round(4),
    'RF Rank':        range(1, len(feature_names) + 1),
    'XGB Importance': xgb_imp.reindex(rf_imp.index).round(4),
    'XGB Rank':       xgb_imp.rank(ascending=False).astype(int).reindex(rf_imp.index)
})
print(f"\nFull Feature Ranking Comparison:\n{rank_df.to_string()}")

# ============================================================
# 1.7: Final Evaluation -- 30 Repeated Train/Test Splits (80/20)
# ============================================================

best_rf_params  = dict(rf_grid.best_params_)
best_xgb_params = dict(xgb_grid.best_params_)

rf_accs, rf_recs, xgb_accs, xgb_recs = [], [], [], []

for seed in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    rf_run = RandomForestClassifier(**best_rf_params, criterion='log_loss', random_state=42, n_jobs=-1)
    rf_run.fit(X_train, y_train)
    y_rf = rf_run.predict(X_test)
    rf_accs.append(accuracy_score(y_test, y_rf))
    rf_recs.append(recall_score(y_test, y_rf))

    xgb_run = XGBClassifier(**best_xgb_params, eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
    xgb_run.fit(X_train, y_train)
    y_xgb = xgb_run.predict(X_test)
    xgb_accs.append(accuracy_score(y_test, y_xgb))
    xgb_recs.append(recall_score(y_test, y_xgb))

print(f"\n{'='*60}")
print("FINAL EVALUATION -- 30 Repeated Train/Test Splits (80/20):")
print(f"   RF  Accuracy: {np.mean(rf_accs):.4f}  ±  {np.std(rf_accs):.4f}")
print(f"   XGB Accuracy: {np.mean(xgb_accs):.4f}  ±  {np.std(xgb_accs):.4f}")
print(f"   RF  Recall:   {np.mean(rf_recs):.4f}  ±  {np.std(rf_recs):.4f}")
print(f"   XGB Recall:   {np.mean(xgb_recs):.4f}  ±  {np.std(xgb_recs):.4f}")

models    = ['Random Forest', 'XGBoost']
acc_means = [np.mean(rf_accs),  np.mean(xgb_accs)]
acc_stds  = [np.std(rf_accs),   np.std(xgb_accs)]
rec_means = [np.mean(rf_recs),  np.mean(xgb_recs)]
rec_stds  = [np.std(rf_recs),   np.std(xgb_recs)]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Final Evaluation: RF vs XGBoost over 30 Train/Test Splits', fontsize=14)

axes[0].bar(models, acc_means, yerr=acc_stds, capsize=10,
            color=['steelblue', 'darkorange'], alpha=0.85, width=0.4)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy (mean ± std)')
axes[0].set_ylim(max(0, min(acc_means) - 0.05), min(1, max(acc_means) + 0.05))
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(models, rec_means, yerr=rec_stds, capsize=10,
            color=['steelblue', 'darkorange'], alpha=0.85, width=0.4)
axes[1].set_ylabel('Recall')
axes[1].set_title('Recall (mean ± std)')
axes[1].set_ylim(max(0, min(rec_means) - 0.1), min(1, max(rec_means) + 0.1))
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
