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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import statsmodels.api as sm

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

# ============================================================
# Q3: PCA in Practice
# ============================================================

workers_path = Path("./Second_Assignment/workers.csv")
workers_df   = pd.read_csv(workers_path)

feat_cols = [
    'age', 'annual_income_gbp', 'standing_hours_per_day', 'lifting_hours_per_day',
    'manual_intensity_score', 'repetitive_motion_score',
    'seated_hours_per_day', 'computer_hours_per_day', 'meetings_hours_per_week'
]
X_w = workers_df[feat_cols]
y_w = workers_df['chronic_pain']

# ============================================================
# 3.1: Data Inspection
# ============================================================

print(f"Workers dataset: {workers_df.shape[0]} observations, {len(feat_cols)} features")
print(f"Chronic pain prevalence: {y_w.mean():.2%}\n")
print("Summary statistics:")
print(X_w.describe().round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(X_w.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Matrix -- Workers Dataset', fontsize=13)
plt.tight_layout()
plt.show()

# ============================================================
# 3.2: PCA -- Standardise and Fit
# ============================================================

# Standardisation is required: annual_income_gbp (~20 000) dwarfs hours-per-day (~1–8);
# PCA maximises variance, so without scaling it would be dominated by income alone.
scaler  = StandardScaler()
X_w_std = scaler.fit_transform(X_w)

pca    = PCA()
X_pca  = pca.fit_transform(X_w_std)

eigvals   = pca.explained_variance_
var_ratio = pca.explained_variance_ratio_
cum_var   = np.cumsum(var_ratio)
n_feat    = len(feat_cols)
pc_labels = [f'PC{i+1}' for i in range(n_feat)]

print(f"\n{'='*60}")
print("PCA -- Variance Explained per Component:")
for i, (ev, vr, cv) in enumerate(zip(eigvals, var_ratio, cum_var)):
    print(f"  PC{i+1}: eigenvalue={ev:.3f}  |  var={vr:.2%}  |  cumulative={cv:.2%}")

# ============================================================
# 3.3: Factor Loadings Plots (Correlation Circles)
# ============================================================

# Correlation loadings: cor(x_i, PC_j) = component[j,i] * sqrt(eigenvalue[j])
# These lie within the unit circle for standardised inputs.
comp_loadings = pca.components_[:3]                         # shape (3, n_features)
corr_loadings = (comp_loadings.T * np.sqrt(eigvals[:3]))    # shape (n_features, 3)
loadings_df   = pd.DataFrame(corr_loadings, index=feat_cols, columns=['PC1', 'PC2', 'PC3'])

def _plot_loading_circle(pc_x, pc_y, ax):
    idx_x = int(pc_x[2:]) - 1
    idx_y = int(pc_y[2:]) - 1
    for feat in feat_cols:
        lx = loadings_df.loc[feat, pc_x]
        ly = loadings_df.loc[feat, pc_y]
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))
        ax.text(lx * 1.12, ly * 1.12, feat, fontsize=7.5, ha='center', va='center')
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    circle = plt.Circle((0, 0), 1, fill=False, color='lightgrey', ls='--')
    ax.add_patch(circle)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel(f'{pc_x}  ({var_ratio[idx_x]:.1%} var)', fontsize=10)
    ax.set_ylabel(f'{pc_y}  ({var_ratio[idx_y]:.1%} var)', fontsize=10)
    ax.set_title(f'Loadings: {pc_x} vs {pc_y}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('PCA: Factor Loadings (Correlation Circles)', fontsize=13)
_plot_loading_circle('PC1', 'PC2', axes[0])
_plot_loading_circle('PC1', 'PC3', axes[1])
_plot_loading_circle('PC2', 'PC3', axes[2])
plt.tight_layout()
plt.show()

# ============================================================
# 3.4: 2D and 3D Projections (coloured by chronic_pain)
# ============================================================

_pain_map = {0: ('steelblue', 'No Chronic Pain'), 1: ('tomato', 'Chronic Pain')}

fig, ax = plt.subplots(figsize=(8, 6))
for label, (color, name) in _pain_map.items():
    mask = y_w == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.5, s=20, label=name)
ax.set_xlabel(f'PC1  ({var_ratio[0]:.1%} var)')
ax.set_ylabel(f'PC2  ({var_ratio[1]:.1%} var)')
ax.set_title('PCA: 2D Projection  (PC1 vs PC2)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 7))
ax3 = fig.add_subplot(111, projection='3d')
for label, (color, name) in _pain_map.items():
    mask = y_w == label
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                c=color, alpha=0.5, s=20, label=name)
ax3.set_xlabel(f'PC1  ({var_ratio[0]:.1%})')
ax3.set_ylabel(f'PC2  ({var_ratio[1]:.1%})')
ax3.set_zlabel(f'PC3  ({var_ratio[2]:.1%})')
ax3.set_title('PCA: 3D Projection  (PC1, PC2, PC3)')
ax3.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 3.5: Dimensionality Reduction Criteria
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PCA: Dimensionality Reduction Criteria', fontsize=13)

# Scree plot
axes[0].plot(range(1, n_feat + 1), eigvals, marker='o', color='steelblue', lw=2)
axes[0].axhline(y=1, color='tomato', ls='--', lw=1.5, label='Kaiser criterion (λ = 1)')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Eigenvalue')
axes[0].set_title('Scree Plot')
axes[0].set_xticks(range(1, n_feat + 1))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, n_feat + 1), cum_var * 100, marker='s', color='darkorange', lw=2)
axes[1].axhline(y=80, color='green',  ls='--', lw=1.5, label='80% threshold')
axes[1].axhline(y=90, color='tomato', ls='--', lw=1.5, label='90% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained (%)')
axes[1].set_title('Cumulative Variance Explained')
axes[1].set_xticks(range(1, n_feat + 1))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

kaiser_n = int(np.sum(eigvals > 1))
pct80_n  = int(np.argmax(cum_var >= 0.80)) + 1
pct90_n  = int(np.argmax(cum_var >= 0.90)) + 1

print(f"\nKaiser criterion (λ > 1):  retain {kaiser_n} component(s)")
print(f"80% variance threshold:    retain {pct80_n} component(s)")
print(f"90% variance threshold:    retain {pct90_n} component(s)")

# ============================================================
# Q3 (cont.): Logistic Regression
# ============================================================

# ============================================================
# 3.6: Data Inspection -- chronic_pain correlations
# ============================================================

pain_corr_orig = X_w.corrwith(y_w).sort_values(key=abs, ascending=False)
print(f"\n{'='*60}")
print("Correlation of original features with chronic_pain:")
print(pain_corr_orig.round(3).to_string())

pc_named = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
pain_corr_pca = pc_named.corrwith(y_w)
print(f"\nCorrelation of retained PCs with chronic_pain:")
print(pain_corr_pca.round(3).to_string())

# ============================================================
# 3.7: Logistic Regression -- Original Variables
# ============================================================

# Full model (all 9 features)
X_orig_full = sm.add_constant(X_w)
logit_full   = sm.Logit(y_w, X_orig_full).fit(disp=0)
print(f"\n{'='*60}")
print("LOGISTIC REGRESSION -- Full model (9 original features):")
print(logit_full.summary2())

# Select significant predictors (p < 0.05, excluding intercept)
sig_feats = [v for v in logit_full.pvalues.index if v != 'const' and logit_full.pvalues[v] < 0.05]
print(f"\nSignificant predictors (p < 0.05): {sig_feats}")

# Refit with significant predictors only
X_orig_sel = sm.add_constant(X_w[sig_feats])
logit_sel   = sm.Logit(y_w, X_orig_sel).fit(disp=0)
print(f"\nLOGISTIC REGRESSION -- Selected model ({len(sig_feats)} features):")
print(logit_sel.summary2())

# Odds ratios + 95% CI for selected model
or_sel    = np.exp(logit_sel.params).drop('const')
or_ci_sel = np.exp(logit_sel.conf_int()).drop('const')
print("\nOdds Ratios (selected model):")
for feat in or_sel.index:
    print(f"  {feat}: OR={or_sel[feat]:.4f}  [{or_ci_sel.loc[feat,0]:.4f}, {or_ci_sel.loc[feat,1]:.4f}]")

# Odds ratio plot -- selected model
fig, ax = plt.subplots(figsize=(8, 5))
y_pos = range(len(or_sel))
ax.barh(list(y_pos), or_sel.values, color='steelblue', alpha=0.8)
for i, feat in enumerate(or_sel.index):
    ax.plot([or_ci_sel.loc[feat, 0], or_ci_sel.loc[feat, 1]], [i, i],
            color='black', lw=2, solid_capstyle='round')
ax.axvline(x=1, color='tomato', ls='--', lw=1.5, label='OR = 1 (no effect)')
ax.set_yticks(list(y_pos))
ax.set_yticklabels(or_sel.index)
ax.set_xlabel('Odds Ratio')
ax.set_title('Odds Ratios -- Selected Original Model (95% CI)')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ============================================================
# 3.8: Logistic Regression -- PCA Variables (3 retained PCs)
# ============================================================

X_pca_3 = sm.add_constant(pc_named)
logit_pca = sm.Logit(y_w, X_pca_3).fit(disp=0)
print(f"\n{'='*60}")
print("LOGISTIC REGRESSION -- PCA model (3 retained PCs):")
print(logit_pca.summary2())

or_pca    = np.exp(logit_pca.params).drop('const')
or_ci_pca = np.exp(logit_pca.conf_int()).drop('const')
print("\nOdds Ratios (PCA model):")
for feat in or_pca.index:
    print(f"  {feat}: OR={or_pca[feat]:.4f}  [{or_ci_pca.loc[feat,0]:.4f}, {or_ci_pca.loc[feat,1]:.4f}]")

# Odds ratio plot -- PCA model
fig, ax = plt.subplots(figsize=(7, 4))
y_pos = range(len(or_pca))
ax.barh(list(y_pos), or_pca.values, color='darkorange', alpha=0.8)
for i, feat in enumerate(or_pca.index):
    ax.plot([or_ci_pca.loc[feat, 0], or_ci_pca.loc[feat, 1]], [i, i],
            color='black', lw=2, solid_capstyle='round')
ax.axvline(x=1, color='tomato', ls='--', lw=1.5, label='OR = 1 (no effect)')
ax.set_yticks(list(y_pos))
ax.set_yticklabels(or_pca.index)
ax.set_xlabel('Odds Ratio')
ax.set_title('Odds Ratios -- PCA Model (3 PCs, 95% CI)')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ============================================================
# 3.9: Model Comparison
# ============================================================

comparison_df = pd.DataFrame({
    'Model':        ['Full original (9)', 'Selected original (6)', 'PCA (3 PCs)'],
    'N Parameters': [int(logit_full.df_model) + 1,
                     int(logit_sel.df_model)  + 1,
                     int(logit_pca.df_model)  + 1],
    'Log-Lik':      [round(logit_full.llf, 2), round(logit_sel.llf, 2), round(logit_pca.llf, 2)],
    'AIC':          [round(logit_full.aic, 2), round(logit_sel.aic, 2), round(logit_pca.aic, 2)],
    'BIC':          [round(logit_full.bic, 2), round(logit_sel.bic, 2), round(logit_pca.bic, 2)],
    'Pseudo R²':    [round(logit_full.prsquared, 4),
                     round(logit_sel.prsquared, 4),
                     round(logit_pca.prsquared, 4)],
})
print(f"\n{'='*60}")
print("MODEL COMPARISON:")
print(comparison_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Logistic Regression Model Comparison', fontsize=13)

models     = comparison_df['Model'].tolist()
aic_vals   = comparison_df['AIC'].tolist()
pr2_vals   = comparison_df['Pseudo R²'].tolist()

axes[0].bar(models, aic_vals, color=['steelblue', 'steelblue', 'darkorange'], alpha=0.85)
axes[0].set_ylabel('AIC')
axes[0].set_title('AIC (lower = better)')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(min(aic_vals) - 5, max(aic_vals) + 5)
for i, v in enumerate(aic_vals):
    axes[0].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=9)

axes[1].bar(models, pr2_vals, color=['steelblue', 'steelblue', 'darkorange'], alpha=0.85)
axes[1].set_ylabel("McFadden's Pseudo R²")
axes[1].set_title("Pseudo R² (higher = better)")
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, max(pr2_vals) + 0.01)
for i, v in enumerate(pr2_vals):
    axes[1].text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.show()
