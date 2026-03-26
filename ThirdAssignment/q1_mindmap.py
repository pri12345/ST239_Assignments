import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# -------------------------------------------------------
# ST239 Mindmap — v2
# Saves q1_mindmap.png (high-res) and q1_mindmap.svg (zoomable)
# -------------------------------------------------------

FIG_W, FIG_H = 72, 48
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(-26, 26)
ax.set_ylim(-17, 13)
ax.axis('off')
fig.patch.set_facecolor('#f5f5f5')
ax.set_facecolor('#f5f5f5')

# -------------------------------------------------------
# Colours
# -------------------------------------------------------
ROOT_C  = '#1a1a2e'
SUP_C   = '#b71c1c'
REG_C   = '#c62828'
CLF_C   = '#d32f2f'
UNSUP_C = '#0d47a1'
DIM_C   = '#1565c0'
CLU_C   = '#1976d2'
FND_C   = '#1b5e20'
EST_C   = '#2e7d32'
EVL_C   = '#388e3c'

# -------------------------------------------------------
# Drawing helpers
# -------------------------------------------------------
def rbox(ax, cx, cy, w, h, fc, ec='white', lw=2.0, radius=0.3, zorder=2, alpha=1.0):
    p = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=zorder, alpha=alpha
    )
    ax.add_patch(p)

def txt(ax, cx, cy, s, fs=9, c='white', bold=False, zorder=5,
        ha='center', va='center'):
    ax.text(cx, cy, s, fontsize=fs, color=c, ha=ha, va=va,
            zorder=zorder, fontweight='bold' if bold else 'normal',
            fontfamily='DejaVu Sans', linespacing=1.4)

def edge(ax, x1, y1, x2, y2, c='#aaaaaa', lw=2.5, zorder=1,
         style='arc3,rad=0.0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-', color=c, lw=lw,
                                connectionstyle=style),
                zorder=zorder)

def dashed(ax, x1, y1, x2, y2, c='#777777', lw=1.4):
    ax.plot([x1, x2], [y1, y2], linestyle='--', color=c,
            lw=lw, zorder=0, alpha=0.65)

def leaf(ax, cx, cy, tag, title, lines, color,
         w=7.5, header_h=1.0, line_h=0.85, pad=0.35, zorder=3):
    """
    Draw a leaf node:
      tag    — short label like '[A]'
      title  — method name (shown in coloured header)
      lines  — list of strings, one per content row
      color  — header colour
    Box height auto-scales to number of lines.
    """
    n = len(lines)
    body_h = n * line_h + 2 * pad
    total_h = header_h + body_h

    # outer white box
    rbox(ax, cx, cy, w, total_h, fc='#ffffff', ec=color,
         lw=2.5, radius=0.25, zorder=zorder)

    # coloured header
    header_top = cy + total_h / 2
    header_cy  = header_top - header_h / 2
    rbox(ax, cx, header_cy, w, header_h, fc=color, ec=color,
         lw=0, radius=0.25, zorder=zorder + 1)
    # tag square in header
    rbox(ax, cx - w/2 + 0.55, header_cy, 0.9, header_h * 0.72,
         fc='white', ec='white', lw=0, radius=0.1, zorder=zorder + 2)
    txt(ax, cx - w/2 + 0.55, header_cy, tag,
        fs=10, c=color, bold=True, zorder=zorder + 3)
    txt(ax, cx + 0.15, header_cy, title,
        fs=11, c='white', bold=True, zorder=zorder + 3)

    # content lines
    body_top = header_top - header_h - pad
    for i, line in enumerate(lines):
        ly = body_top - (i + 0.5) * line_h
        txt(ax, cx, ly, line, fs=10.5, c='#222222', zorder=zorder + 2)

    # return bottom centre for edge connection
    return cy - total_h / 2

def branch_node(ax, cx, cy, title, subtitle, color, w=4.5, h=1.2, zorder=3):
    rbox(ax, cx, cy, w, h, fc=color, ec='white', lw=2.5,
         radius=0.3, zorder=zorder)
    txt(ax, cx, cy + 0.2, title,  fs=11, bold=True, zorder=zorder+1)
    txt(ax, cx, cy - 0.3, subtitle, fs=8, c='#eeeeee', zorder=zorder+1)

def sub_node(ax, cx, cy, title, subtitle, color, w=4.0, h=1.0, zorder=3):
    rbox(ax, cx, cy, w, h, fc=color, ec='white', lw=2.0,
         radius=0.25, zorder=zorder)
    txt(ax, cx, cy + 0.17, title,    fs=10, bold=True, zorder=zorder+1)
    txt(ax, cx, cy - 0.22, subtitle, fs=7.5, c='#eeeeee', zorder=zorder+1)

# -------------------------------------------------------
# ROOT
# -------------------------------------------------------
rbox(ax, 0, 0, 5.5, 1.5, fc=ROOT_C, ec='white', lw=3, radius=0.45, zorder=4)
txt(ax, 0,  0.25, 'ST239',                        fs=17, bold=True, zorder=5)
txt(ax, 0, -0.32, 'Statistical & ML Methods',     fs=10, c='#bbbbbb', zorder=5)

# -------------------------------------------------------
# BRANCH 1 — SUPERVISED LEARNING  (left)
# -------------------------------------------------------
SX, SY = -11.0, 2.0
branch_node(ax, SX, SY, 'Supervised Learning',
            'learn f(X)→Y from labelled data', SUP_C, w=5.5)
edge(ax, -2.75, 0.3, SX + 2.75, SY, c=SUP_C, lw=3)

# -- Regression sub-branch
RX, RY = -16.0, 6.5
sub_node(ax, RX, RY, 'Regression', 'continuous Y', REG_C, w=3.8)
edge(ax, SX - 2.75, SY + 0.35, RX + 1.9, RY - 0.35,
     c=REG_C, lw=2.5, style='arc3,rad=0.12')

LW = 7.8   # leaf width (left side)

leaf(ax, -22.5, 8.5, '[A]', 'Simple Linear Regression',
     ['Key eq.:  Y = β₀ + β₁X + ε',
      'Fit:  OLS — minimise RSS, closed-form solution',
      'β̂₁ = Cov(X,Y) / Var(X)'],
     REG_C, w=LW)
edge(ax, RX - 1.9, RY + 0.1, -22.5 + LW/2, 8.5 - 0.4,
     c=REG_C, lw=2)

leaf(ax, -22.5, 5.8, '[B]', 'Multiple Linear Regression',
     ['Key eq.:  β̂ = (XᵀX)⁻¹XᵀY  (matrix form)',
      'Fit:  OLS, p predictors, dummy encoding for categoricals',
      'Model selection:  Adj. R², AIC/BIC, stepwise, CV'],
     REG_C, w=LW)
edge(ax, RX - 1.9, RY - 0.1, -22.5 + LW/2, 5.8 + 0.4,
     c=REG_C, lw=2)

# -- Classification sub-branch
CX, CY = -16.0, -1.5
sub_node(ax, CX, CY, 'Classification', 'discrete Y', CLF_C, w=3.8)
edge(ax, SX - 2.75, SY - 0.35, CX + 1.9, CY + 0.35,
     c=CLF_C, lw=2.5, style='arc3,rad=-0.1')

leaf(ax, -22.5, 2.2, '[C]', 'Logistic Regression',
     ['Key eq.:  log(p / 1−p) = Xβ  →  p = sigmoid(Xβ)',
      'Fit:  MLE (Bernoulli likelihood), iterative (Newton-Raphson)',
      'Interpretation:  exp(βⱼ) = odds ratio per unit increase in Xⱼ'],
     CLF_C, w=LW)
edge(ax, CX - 1.9, CY + 0.25, -22.5 + LW/2, 2.2 - 0.4,
     c=CLF_C, lw=2)

leaf(ax, -22.5, -0.6, '[D]', 'Decision Trees',
     ['Split criterion:  Gini = Σpₖ(1−pₖ)  (clf)  /  RSS (reg)',
      'Fit:  recursive partitioning; pruning controls overfitting',
      'Leaf = majority class (clf) or mean (reg)'],
     CLF_C, w=LW)
edge(ax, CX - 1.9, CY, -22.5 + LW/2, -0.6 + 0.4,
     c=CLF_C, lw=2)

leaf(ax, -22.5, -3.4, '[E]', 'Random Forest',
     ['Idea:  B trees on bootstrap samples, m random features/split',
      'Variance ↓ by averaging decorrelated trees (low bias, low var)',
      'Feature importance:  Mean Decrease in Impurity (MDI)'],
     CLF_C, w=LW)
edge(ax, CX - 1.9, CY - 0.15, -22.5 + LW/2, -3.4 + 0.4,
     c=CLF_C, lw=2)

leaf(ax, -22.5, -6.2, "[E']", 'XGBoost',
     ['Key eq.:  F_t(x) = F_{t-1}(x) + η · h_t(x)',
      'Fit:  boosting — sequential trees each correcting residuals',
      'Hyperparams:  learning rate η, depth, n_estimators, subsample'],
     CLF_C, w=LW)
edge(ax, CX - 1.9, CY - 0.3, -22.5 + LW/2, -6.2 + 0.4,
     c=CLF_C, lw=2)

leaf(ax, -22.5, -9.0, '[SVM]', 'Support Vector Machine',
     ['Key eq.:  min ‖w‖²  s.t.  yᵢ(wᵀxᵢ + b) ≥ 1',
      'Soft margin:  C penalises misclassification (large C = tight fit)',
      'Kernel trick:  RBF K = exp(−γ‖x−x′‖²), QP solver'],
     CLF_C, w=LW)
edge(ax, CX - 1.9, CY - 0.4, -22.5 + LW/2, -9.0 + 0.4,
     c=CLF_C, lw=2)

# -------------------------------------------------------
# BRANCH 2 — UNSUPERVISED LEARNING  (right)
# -------------------------------------------------------
UX, UY = 11.0, 2.0
branch_node(ax, UX, UY, 'Unsupervised Learning',
            'find hidden structure, no labels', UNSUP_C, w=5.5)
edge(ax, 2.75, 0.3, UX - 2.75, UY, c=UNSUP_C, lw=3)

# -- Dimensionality Reduction
DX, DY = 16.5, 6.5
sub_node(ax, DX, DY, 'Dimensionality Reduction', 'reduce complexity', DIM_C, w=4.2)
edge(ax, UX + 2.75, UY + 0.35, DX - 2.1, DY - 0.35,
     c=DIM_C, lw=2.5, style='arc3,rad=-0.12')

LWR = 7.8   # leaf width (right side)

leaf(ax, 23.0, 6.5, '[F]', 'Principal Component Analysis',
     ['Key eq.:  Σvᵢ = λᵢvᵢ  (eigendecomposition of covariance matrix)',
      'Procedure:  standardise → eigendecompose → project onto top-k PCs',
      'Choose k:  scree plot elbow / cumulative variance ≥ 80%'],
     DIM_C, w=LWR)
edge(ax, DX + 2.1, DY, 23.0 - LWR/2, 6.5 + 0.1,
     c=DIM_C, lw=2)

# -- Clustering
KX, KY = 16.5, -1.5
sub_node(ax, KX, KY, 'Clustering', 'group similar observations', CLU_C, w=4.2)
edge(ax, UX + 2.75, UY - 0.35, KX - 2.1, KY + 0.35,
     c=CLU_C, lw=2.5, style='arc3,rad=0.1')

leaf(ax, 23.0, 2.2, '[G]', 'Hierarchical Clustering',
     ['Idea:  build dendrogram by iteratively merging clusters',
      'Linkage:  Ward — minimise within-cluster variance at each merge',
      'Cut dendrogram at chosen height → k clusters (no need to prespecify k)'],
     CLU_C, w=LWR)
edge(ax, KX + 2.1, KY + 0.25, 23.0 - LWR/2, 2.2 - 0.4,
     c=CLU_C, lw=2)

leaf(ax, 23.0, -0.6, '[H]', 'K-Means',
     ['Key eq.:  WCSS = Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖²',
      'Fit:  Lloyd\'s — assign points → update centroids → repeat',
      'Choose k:  elbow plot (WCSS vs k) + silhouette score'],
     CLU_C, w=LWR)
edge(ax, KX + 2.1, KY, 23.0 - LWR/2, -0.6 + 0.4,
     c=CLU_C, lw=2)

leaf(ax, 23.0, -3.4, '[I]', 'DBSCAN',
     ['Idea:  dense regions separated by low-density areas',
      'Params:  ε (neighbourhood radius), min_samples (density threshold)',
      'Point types:  core / border / noise (labelled −1)'],
     CLU_C, w=LWR)
edge(ax, KX + 2.1, KY - 0.3, 23.0 - LWR/2, -3.4 + 0.4,
     c=CLU_C, lw=2)

# -------------------------------------------------------
# BRANCH 3 — STATISTICAL FOUNDATIONS  (bottom)
# -------------------------------------------------------
FX, FY = 0.0, -4.5
branch_node(ax, FX, FY, 'Statistical Foundations',
            'cross-cutting tools — all branches', FND_C, w=5.8)
edge(ax, 0, -0.75, FX, FY + 0.6, c=FND_C, lw=3)

LWF = 7.2   # leaf width (foundations)

# -- Parameter Estimation
EX, EY = -7.5, -8.5
sub_node(ax, EX, EY, 'Parameter Estimation', 'how are params fitted?', EST_C, w=4.0)
edge(ax, FX - 2.9, FY - 0.5, EX + 2.0, EY + 0.35,
     c=EST_C, lw=2.5, style='arc3,rad=0.15')

leaf(ax, -13.5, -12.0, '[J]', 'OLS — Ordinary Least Squares',
     ['Key eq.:  β̂ = (XᵀX)⁻¹XᵀY',
      'Criterion:  minimise RSS = Σ(yᵢ − ŷᵢ)²',
      'Closed-form solution — applies to SLR and MLR',
      'OLS = MLE under Gaussian errors'],
     EST_C, w=LWF)
edge(ax, EX - 2.0, EY - 0.3, -13.5 + LWF/2, -12.0 + 0.4,
     c=EST_C, lw=2)

leaf(ax, -4.5, -12.0, '[K]', 'MLE — Maximum Likelihood Estimation',
     ['Key eq.:  θ̂ = argmax Σ log f(yᵢ | θ)',
      'Fit:  iterative numerical optimisation (Newton-Raphson)',
      'Requires distributional assumption on the data',
      'Used in:  Logistic Regression (Bernoulli likelihood)'],
     EST_C, w=LWF)
edge(ax, EX + 0.5, EY - 0.45, -4.5 - LWF/2 + 0.5, -12.0 + 0.4,
     c=EST_C, lw=2)

# -- Model Evaluation
VX, VY = 7.5, -8.5
sub_node(ax, VX, VY, 'Model Evaluation', 'how do we select models?', EVL_C, w=4.0)
edge(ax, FX + 2.9, FY - 0.5, VX - 2.0, VY + 0.35,
     c=EVL_C, lw=2.5, style='arc3,rad=-0.15')

leaf(ax, 4.5, -12.0, '[L]', 'In-Sample Criteria',
     ['R²:  proportion of variance explained = 1 − RSS/TSS',
      'Adj. R²:  penalises extra predictors (use over plain R²)',
      'AIC = −2 log L + 2k   |   BIC = −2 log L + log(n)·k',
      'Lower AIC/BIC = better; BIC stricter for large n'],
     EVL_C, w=LWF)
edge(ax, VX - 0.8, VY - 0.45, 4.5 + LWF/2 - 0.5, -12.0 + 0.4,
     c=EVL_C, lw=2)

leaf(ax, 13.5, -12.0, '[M]', 'Cross-Validation & Classification Metrics',
     ['K-fold CV:  train on k−1 folds, test on 1, rotate all folds',
      'Stratified CV:  preserves class balance in each fold',
      'Accuracy, Precision, Recall, F1 = 2PR/(P+R)',
      'AUC-ROC:  area under TPR vs FPR curve (0.5=random, 1=perfect)'],
     EVL_C, w=LWF)
edge(ax, VX + 2.0, VY - 0.3, 13.5 - LWF/2, -12.0 + 0.4,
     c=EVL_C, lw=2)

# -------------------------------------------------------
# Cross-branch dotted connections
# -------------------------------------------------------
# OLS ↔ MLE
dashed(ax, -13.5 + LWF/2, -12.0, -4.5 - LWF/2, -12.0, c='#555555', lw=1.6)
ax.text(-9.0, -12.6, 'OLS = MLE under Gaussian errors',
        fontsize=8, ha='center', color='#444444', style='italic', zorder=5)

# LogReg ↔ MLE
dashed(ax, -22.5 + LWF/2, 2.2 - 1.5, -4.5 - LWF/2 + 1.5, -12.0 + 1.5,
       c='#888888', lw=1.2)
ax.text(-14.5, -5.0, 'LogReg fitted\nvia MLE',
        fontsize=7.5, ha='center', color='#666666', style='italic', zorder=5,
        bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#cccccc', alpha=0.85))

# PCA → Clustering
dashed(ax, 23.0 - LWF/2, 6.5 - 1.5, KX + 2.1, KY + 0.4,
       c=DIM_C, lw=1.3)
ax.text(21.0, 3.8, 'PCA often used as\npreprocessing for clustering',
        fontsize=7.5, ha='center', color=DIM_C, style='italic', zorder=5,
        bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#bbdefb', alpha=0.85))

# Decision Trees → RF & XGBoost
ax.text(-24.5, -4.8, 'RF & XGBoost\nbuild on trees ↑',
        fontsize=7.5, ha='center', color=CLF_C, style='italic', zorder=5,
        bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#ffcdd2', alpha=0.85))

# -------------------------------------------------------
# Title
# -------------------------------------------------------
ax.text(0, 12.2, 'ST239 Mindmap - Student ID 5558899',
        fontsize=22, fontweight='bold', ha='center', color='#1a1a2e', zorder=5)

plt.tight_layout(pad=0.3)

out_png = 'ThirdAssignment/q1_mindmap.png'
out_svg = 'ThirdAssignment/q1_mindmap.svg'
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='#f5f5f5')
plt.savefig(out_svg, bbox_inches='tight', facecolor='#f5f5f5')
print(f"Saved: {out_png}")
print(f"Saved: {out_svg}  (open in browser for infinite zoom)")
plt.show()
