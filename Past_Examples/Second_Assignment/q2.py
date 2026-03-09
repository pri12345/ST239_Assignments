import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

# ==========================================
# Question 2 Data Initialization
# ==========================================
data = {
    'ID': list(range(1, 26)),
    'TrainingHours': [10, 9, 8, 11, 7, 6, 12, 5, 4, 8, 9, 3, 2, 6, 7, 10, 5, 4, 8, 9, 11, 3, 12, 7, 6],
    'FitnessPass': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    'AttendanceGood': [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    'PriorClub': [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    'TopTeam': [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

def calc_gini(counts):
    total = sum(counts)
    if total == 0:
        return 0
    probs = [c / total for c in counts]
    return 1 - sum(p**2 for p in probs)

# ==========================================
# Part 1: Gini Impurity at Root Node
# ==========================================
counts = df['TopTeam'].value_counts()
class_0 = counts.get(0, 0)
class_1 = counts.get(1, 0)
gini_root = calc_gini([class_0, class_1])
print(f"--- Part 1: Gini Root ---")
print(f"Total samples: {len(df)}")
print(f"Class 0 (No Top Team): {class_0}")
print(f"Class 1 (Top Team): {class_1}")
print(f"Gini root: {gini_root:.4f}\n")

# ==========================================
# Part 3: First Split on TrainingHours
# ==========================================
# Coach requires >= 4 sessions of 2 hours = 8 hours
split_cond = df['TrainingHours'] >= 8

# Left child: < 8
left_child = df[~split_cond]
left_counts = left_child['TopTeam'].value_counts()
gini_left = calc_gini([left_counts.get(0, 0), left_counts.get(1, 0)])

# Right child: >= 8
right_child = df[split_cond]
right_counts = right_child['TopTeam'].value_counts()
gini_right = calc_gini([right_counts.get(0, 0), right_counts.get(1, 0)])

# Weighted Gini
n_left = len(left_child)
n_right = len(right_child)
n_total = len(df)
gini_weighted = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

print(f"--- Part 3: First Split (TrainingHours >= 8) ---")
print(f"Left Child (< 8): {n_left} samples, Gini: {gini_left:.4f}")
print(f"Right Child (>= 8): {n_right} samples, Gini: {gini_right:.4f}")
print(f"Weighted Gini: {gini_weighted:.4f}")
print(f"Impurity reduction: {gini_root - gini_weighted:.4f}\n")

# ==========================================
# Part 2: DecisionTreeClassifier max_depth=2
# ==========================================
X = df[['TrainingHours', 'FitnessPass', 'AttendanceGood', 'PriorClub']]
y = df['TopTeam']

clf_depth_2 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=2, random_state=42)
clf_depth_2.fit(X, y)

print(f"--- Part 2: DecisionTreeClassifier Output (max_depth=2) ---")
tree_rules = export_text(clf_depth_2, feature_names=list(X.columns))
print(tree_rules)

# ==========================================
# Part 4: Second-Level Split Evaluation
# ==========================================
# We need to find the best split for the left child (< 8) and right child (>= 8)
# Left Child (< 8)
print(f"--- Part 4: Second Level Splits Evaluation (Left Child: TrainingHours < 8) ---")
best_gini_left = 1.0
best_var_left = None

for var in ['FitnessPass', 'AttendanceGood', 'PriorClub']:
    val_0 = left_child[left_child[var] == 0]
    val_1 = left_child[left_child[var] == 1]
    
    counts_0 = val_0['TopTeam'].value_counts()
    counts_1 = val_1['TopTeam'].value_counts()
    
    g0 = calc_gini([counts_0.get(0, 0), counts_0.get(1, 0)])
    g1 = calc_gini([counts_1.get(0, 0), counts_1.get(1, 0)])
    
    gw = (len(val_0) / n_left) * g0 + (len(val_1) / n_left) * g1
    print(f"Split on {var}: Weighted Gini = {gw:.4f}")
    
    if gw < best_gini_left:
        best_gini_left = gw
        best_var_left = var

print(f"Best split for Left Child is on {best_var_left} (Gini={best_gini_left:.4f})\n")

# Right Child (>= 8)
print(f"--- Part 4: Second Level Splits Evaluation (Right Child: TrainingHours >= 8) ---")
best_gini_right = 1.0
best_var_right = None

for var in ['FitnessPass', 'AttendanceGood', 'PriorClub']:
    val_0 = right_child[right_child[var] == 0]
    val_1 = right_child[right_child[var] == 1]
    
    counts_0 = val_0['TopTeam'].value_counts()
    counts_1 = val_1['TopTeam'].value_counts()
    
    g0 = calc_gini([counts_0.get(0, 0), counts_0.get(1, 0)])
    g1 = calc_gini([counts_1.get(0, 0), counts_1.get(1, 0)])
    
    gw = (len(val_0) / n_right) * g0 + (len(val_1) / n_right) * g1
    print(f"Split on {var}: Weighted Gini = {gw:.4f}")
    
    if gw < best_gini_right:
        best_gini_right = gw
        best_var_right = var

print(f"Best split for Right Child is on {best_var_right} (Gini={best_gini_right:.4f})\n")

# ==========================================
# Question 2: Student Predictions (Shallow Tree)
# ==========================================
# Hand-made tree rules:
# Root: split by TrainingHours >= 8
# Left child: split by PriorClub == 1
# Right child: split by FitnessPass == 1
print("--- Question 2: Student Predictions (Shallow Tree) ---")
students = pd.DataFrame([
    {'Student': 'A', 'TrainingHours': 9, 'FitnessPass': 0, 'AttendanceGood': 1, 'PriorClub': 1},
    {'Student': 'B', 'TrainingHours': 6, 'FitnessPass': 1, 'AttendanceGood': 1, 'PriorClub': 0}
])

for idx, row in students.iterrows():
    path = "Root -> "
    if row['TrainingHours'] >= 8:
        path += "Right Child (TrainingHours >= 8) -> "
        if row['FitnessPass'] == 1:
            path += "Right Leaf (FitnessPass == 1) -> Prediction: Class 1"
        else:
            path += "Left Leaf (FitnessPass == 0) -> Prediction: Class 0"
    else:
        path += "Left Child (TrainingHours < 8) -> "
        if row['PriorClub'] == 1:
            path += "Right Leaf (PriorClub == 1) -> Prediction: Class 0"
        else:
            path += "Left Leaf (PriorClub == 0) -> Prediction: Class 0"
    print(f"Student {row['Student']}: {path}")

print("\n")

# ==========================================
# Max Depth Tree
# ==========================================
# Considering all variables and reaching max depth possible
clf_max = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, random_state=42)
clf_max.fit(X, y)

print(f"--- Max Depth Tree Output ---")
tree_rules_max = export_text(clf_max, feature_names=list(X.columns))
print(tree_rules_max)

print("--- Max Depth Tree Properties ---")
print(f"Number of leaves: {clf_max.get_n_leaves()}")
print(f"Depth of tree: {clf_max.get_depth()}")
print(f"Classes predicted based on output of decision tree predict:")
print(f"Student A Predict: {clf_max.predict(students[X.columns].iloc[[0]])[0]}")
print(f"Student B Predict: {clf_max.predict(students[X.columns].iloc[[1]])[0]}")


