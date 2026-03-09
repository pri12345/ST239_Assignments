# Assignment 2 — Question 2: 'Planting' Your Decision Tree (Handwritten Mockup)

> **What this file is:** A complete worked solution with all calculations and explanations,
> organised exactly in the order the question asks. Each section opens with a grey-quoted
> block explaining what the question wants; what follows is what you write/draw on paper.

---

## Dataset Reference

| ID | TH | FP | AG | PC | TT | | ID | TH | FP | AG | PC | TT |
|----|----|----|----|----|----|---|----|----|----|----|----|----|
| 1  | 10 | 1  | 1  | 1  | 1  | | 14 | 6  | 1  | 0  | 1  | 0  |
| 2  | 9  | 1  | 1  | 0  | 1  | | 15 | 7  | 0  | 1  | 0  | 0  |
| 3  | 8  | 1  | 0  | 1  | 1  | | 16 | 10 | 1  | 0  | 1  | 1  |
| 4  | 11 | 1  | 1  | 1  | 1  | | 17 | 5  | 1  | 1  | 0  | 0  |
| 5  | 7  | 1  | 1  | 1  | 1  | | 18 | 4  | 1  | 0  | 0  | 0  |
| 6  | 6  | 1  | 1  | 0  | 0  | | 19 | 8  | 1  | 1  | 0  | 1  |
| 7  | 12 | 1  | 1  | 0  | 1  | | 20 | 9  | 0  | 0  | 1  | 0  |
| 8  | 5  | 0  | 1  | 1  | 0  | | 21 | 11 | 1  | 1  | 1  | 1  |
| 9  | 4  | 0  | 1  | 0  | 0  | | 22 | 3  | 1  | 1  | 0  | 0  |
| 10 | 8  | 0  | 1  | 1  | 1  | | 23 | 12 | 1  | 0  | 1  | 1  |
| 11 | 9  | 1  | 0  | 0  | 1  | | 24 | 7  | 1  | 0  | 0  | 0  |
| 12 | 3  | 0  | 0  | 0  | 0  | | 25 | 6  | 0  | 1  | 1  | 0  |
| 13 | 2  | 0  | 0  | 1  | 0  |

**Key:** TH = TrainingHours, FP = FitnessPass, AG = AttendanceGood, PC = PriorClub, TT = TopTeam (target)

**Class totals:**
- TopTeam = 1: IDs {1,2,3,4,5,7,10,11,16,19,21,23} → **12 students**
- TopTeam = 0: IDs {6,8,9,12,13,14,15,17,18,20,22,24,25} → **13 students**

---

## Part 1: Gini Impurity at the Root Node

> **What the question asks:** Write down the Gini formula and compute it for all 25 observations
> (the root node = the whole dataset, before any split).

**Gini impurity formula:**

$$\text{Gini}(t) = 1 - \sum_{k} p_k^2$$

where $p_k$ is the fraction of class $k$ in node $t$.

**At the root (n = 25):**

$$p(\text{TT}=1) = \frac{12}{25} = 0.48, \qquad p(\text{TT}=0) = \frac{13}{25} = 0.52$$

$$\text{Gini}_{\text{root}} = 1 - (0.48^2 + 0.52^2) = 1 - (0.2304 + 0.2704) = 1 - 0.5008 = \boxed{0.4992}$$

> **Why 0.4992?** Gini = 0 means perfectly pure (all one class). Gini = 0.5 is the maximum
> impurity for a two-class problem (50/50 split). At 0.4992 the root is nearly maximally impure —
> the classes are almost balanced, so the tree has a lot of sorting work ahead.

---

## Part 2: What Would the Sklearn Code Produce? (max_depth=2)

> **What the question asks:** Explain what the output of
> `DecisionTreeClassifier(DATA, criterion='gini', splitter='best', max_depth=2)`
> would look like as a decision tree flowchart.

**Explanation:**

- `criterion='gini'` — at each node the algorithm tries every possible split on every variable
  and picks the one that minimises the weighted Gini of the two resulting children.
- `splitter='best'` — exhaustive search; no randomness (unlike a random forest).
- `max_depth=2` — the tree can have at most 2 layers of splits, giving at most 4 leaf nodes.

**Which variable would sklearn pick as root?** To answer this we compare the weighted Gini
for a single split on each variable across all 25 students:

| Variable | Weighted Gini after split |
|----------|--------------------------|
| **TrainingHours** (≥ 8 vs < 8) | **0.1472** ← lowest → best |
| PriorClub | 0.4595 |
| FitnessPass | 0.3806 |
| AttendanceGood | 0.4907 |

TrainingHours gives by far the largest impurity reduction, so **sklearn naturally chooses
TrainingHours as root** — no forcing needed. As we show in Part 4, it then independently chooses
FitnessPass for the right branch and PriorClub for the left branch. The sklearn depth-2 output
therefore matches exactly the tree we build by hand below.

**Flowchart description of the sklearn output** (detailed numbers in Part 5):

```
Root: TrainingHours <= 7.5
   Left (TH < 8):  split on PriorClub <= 0.5
      → 2 leaves (both predict class 0)
   Right (TH >= 8): split on FitnessPass <= 0.5
      → 2 leaves (predict class 0 and class 1)
```

---

## Part 3: Forced First Split on TrainingHours (≥ 8 hours/week)

> **What the question asks:** The coach requires 4 sessions × 2 hours = **8 hours/week** minimum.
> Force TH = 8 as the first split. Show:
> 1. Which athlete IDs go into each branch.
> 2. Gini impurity for each child node.
> 3. Weighted combined Gini.
> 4. Impurity reduction.

**Split condition:** TrainingHours ≥ 8 (Right) vs TrainingHours < 8 (Left)

---

### Right branch: TH ≥ 8 → n = 12

IDs: {1, 2, 3, 4, 7, 10, 11, 16, 19, 20, 21, 23}

| Class | IDs | Count |
|-------|-----|-------|
| TT = 1 | 1,2,3,4,7,10,11,16,19,21,23 | 11 |
| TT = 0 | 20 | 1 |

$$p_1 = \frac{11}{12}, \quad p_0 = \frac{1}{12}$$

$$\text{Gini}_{\text{right}} = 1 - \left(\frac{11}{12}\right)^2 - \left(\frac{1}{12}\right)^2 = 1 - \frac{121}{144} - \frac{1}{144} = 1 - \frac{122}{144} = \frac{22}{144} \approx \mathbf{0.1528}$$

---

### Left branch: TH < 8 → n = 13

IDs: {5, 6, 8, 9, 12, 13, 14, 15, 17, 18, 22, 24, 25}

| Class | IDs | Count |
|-------|-----|-------|
| TT = 1 | 5 | 1 |
| TT = 0 | 6,8,9,12,13,14,15,17,18,22,24,25 | 12 |

$$p_1 = \frac{1}{13}, \quad p_0 = \frac{12}{13}$$

$$\text{Gini}_{\text{left}} = 1 - \left(\frac{1}{13}\right)^2 - \left(\frac{12}{13}\right)^2 = 1 - \frac{1}{169} - \frac{144}{169} = \frac{24}{169} \approx \mathbf{0.1420}$$

---

### Weighted Gini after this split:

$$\text{Gini}_{\text{weighted}} = \frac{12}{25} \times 0.1528 + \frac{13}{25} \times 0.1420 = 0.0733 + 0.0738 = \mathbf{0.1472}$$

### Impurity Reduction:

$$\Delta\text{Gini} = 0.4992 - 0.1472 = \mathbf{0.3520}$$

> **Interpretation:** A reduction of 0.352 is very large. The right branch is nearly pure
> (11/12 selected), and the left is nearly pure (12/13 not selected). TrainingHours is a powerful
> separator — athletes who train enough are almost all selected; those who don't are almost all not.

---

## Part 4: Best Second-Level Split (all three scenarios, both branches)

> **What the question asks:** For each of the two branches, evaluate all 3 remaining binary
> variables (FitnessPass, AttendanceGood, PriorClub) and pick the best split. The question
> explicitly says to show "all three different scenarios".

---

### RIGHT branch (TH ≥ 8, n = 12): {1,2,3,4,7,10,11,16,19,20,21,23}

#### Scenario A — FitnessPass

| Group | IDs | TT=1 | TT=0 | n | Gini |
|-------|-----|------|------|---|------|
| FP = 1 | 1,2,3,4,7,11,16,19,21,23 | 10 | 0 | 10 | $1 - 1^2 - 0^2 = \mathbf{0.0}$ (pure) |
| FP = 0 | 10, 20 | 1 | 1 | 2 | $1 - 0.5^2 - 0.5^2 = \mathbf{0.5}$ |

$$\text{Gini}_{\text{FP} \mid \text{right}} = \frac{10}{12}(0) + \frac{2}{12}(0.5) = \mathbf{0.0833}$$

#### Scenario B — AttendanceGood

| Group | IDs | TT=1 | TT=0 | n | Gini |
|-------|-----|------|------|---|------|
| AG = 1 | 1,2,4,7,10,19,21 | 7 | 0 | 7 | $\mathbf{0.0}$ (pure) |
| AG = 0 | 3,11,16,20,23 | 4 | 1 | 5 | $1 - (4/5)^2 - (1/5)^2 = 8/25 = \mathbf{0.32}$ |

$$\text{Gini}_{\text{AG} \mid \text{right}} = \frac{7}{12}(0) + \frac{5}{12}(0.32) = \mathbf{0.1333}$$

#### Scenario C — PriorClub

| Group | IDs | TT=1 | TT=0 | n | Gini |
|-------|-----|------|------|---|------|
| PC = 1 | 1,3,4,10,16,20,21,23 | 7 | 1 | 8 | $1-(7/8)^2-(1/8)^2 = 14/64 \approx \mathbf{0.2188}$ |
| PC = 0 | 2,7,11,19 | 4 | 0 | 4 | $\mathbf{0.0}$ (pure) |

$$\text{Gini}_{\text{PC} \mid \text{right}} = \frac{8}{12}(0.2188) + \frac{4}{12}(0) = \mathbf{0.1458}$$

#### Summary — Right branch:

| Variable | Weighted Gini | Verdict |
|----------|--------------|---------|
| **FitnessPass** | **0.0833** | **Best — choose this** |
| AttendanceGood | 0.1333 | |
| PriorClub | 0.1458 | |

---

### LEFT branch (TH < 8, n = 13): {5,6,8,9,12,13,14,15,17,18,22,24,25}

#### Scenario A — FitnessPass

| Group | IDs | TT=1 | TT=0 | n | Gini |
|-------|-----|------|------|---|------|
| FP = 1 | 5,6,14,17,18,22,24 | 1 | 6 | 7 | $1-(1/7)^2-(6/7)^2 = 12/49 \approx \mathbf{0.2449}$ |
| FP = 0 | 8,9,12,13,15,25 | 0 | 6 | 6 | $\mathbf{0.0}$ (pure) |

$$\text{Gini}_{\text{FP} \mid \text{left}} = \frac{7}{13}(0.2449) + \frac{6}{13}(0) = \mathbf{0.1319}$$

#### Scenario B — AttendanceGood

| Group | IDs | TT=1 | TT=0 | n | Gini |
|-------|-----|------|------|---|------|
| AG = 1 | 5,6,8,9,15,17,22,25 | 1 | 7 | 8 | $1-(1/8)^2-(7/8)^2 = 14/64 \approx \mathbf{0.2188}$ |
| AG = 0 | 12,13,14,18,24 | 0 | 5 | 5 | $\mathbf{0.0}$ (pure) |

$$\text{Gini}_{\text{AG} \mid \text{left}} = \frac{8}{13}(0.2188) + \frac{5}{13}(0) = \mathbf{0.1346}$$

#### Scenario C — PriorClub

| Group | IDs | TT=1 | TT=0 | n | Gini |
|-------|-----|------|------|---|------|
| PC = 1 | 5,8,13,14,25 | 1 | 4 | 5 | $1-(1/5)^2-(4/5)^2 = 8/25 = \mathbf{0.32}$ |
| PC = 0 | 6,9,12,15,17,18,22,24 | 0 | 8 | 8 | $\mathbf{0.0}$ (pure) |

$$\text{Gini}_{\text{PC} \mid \text{left}} = \frac{5}{13}(0.32) + \frac{8}{13}(0) = \mathbf{0.1231}$$

#### Summary — Left branch:

| Variable | Weighted Gini | Verdict |
|----------|--------------|---------|
| FitnessPass | 0.1319 | |
| AttendanceGood | 0.1346 | |
| **PriorClub** | **0.1231** | **Best — choose this** |

> **Justification:** We choose the variable that minimises the weighted Gini at each node independently.
> FitnessPass is best for the Right branch; PriorClub is best for the Left branch.

---

## Part 5: Final Depth-2 Tree — Flow Diagram

> **What the question asks:** Write the flowchart showing: root node + starting Gini, first
> split, intermediate nodes with Gini, the second splits, and leaf Gini/predictions — as
> if it were a Python `export_text` output.

```
|--- TrainingHours <= 7.50        [Root:  n=25, Gini=0.4992]
|   |--- PriorClub <= 0.50        [Node:  n=13, Gini=0.1420]
|   |   |--- class: 0             [Leaf:  n=8,  Gini=0.0000]  IDs: 6,9,12,15,17,18,22,24
|   |   |--- class: 0             [Leaf:  n=5,  Gini=0.3200]  IDs: 5,8,13,14,25  (1 vs 4, majority=0)
|--- TrainingHours >  7.50        [Node:  n=12, Gini=0.1528]
|   |--- FitnessPass <= 0.50      [Node:  n=12, Gini=0.1528]
|   |   |--- class: 0             [Leaf:  n=2,  Gini=0.5000]  IDs: 10,20  (1 vs 1, tied → predict 0)
|   |   |--- class: 1             [Leaf:  n=10, Gini=0.0000]  IDs: 1,2,3,4,7,11,16,19,21,23
```

**As a diagram:**

```
                     [Root] n=25, Gini=0.4992
                   TrainingHours <= 7.5?
                   /                      \
              YES (TH < 8)            NO (TH >= 8)
            n=13, Gini=0.1420        n=12, Gini=0.1528
                  |                         |
          PriorClub = 0?              FitnessPass = 0?
          /           \               /             \
        YES            NO           YES              NO
   [Leaf] n=8      [Leaf] n=5   [Leaf] n=2      [Leaf] n=10
   Gini=0.00       Gini=0.32    Gini=0.50        Gini=0.00
   Predict: 0      Predict: 0   Predict: 0       Predict: 1
```

**Notes on the leaves:**
- PC=0 leaf: all 8 athletes with TH<8 and no prior club are not selected → pure → Gini=0.
- PC=1 leaf: 1 selected, 4 not → majority class = 0 → Gini = 8/25 = 0.32. Not pure, but tree stops.
- FP=0 leaf: IDs 10 (TT=1) and 20 (TT=0) → tie → sklearn predicts 0 (lowest label) → Gini=0.5.
- FP=1 leaf: all 10 selected → pure → Gini=0.

**Overall weighted Gini of depth-2 tree:**

$$\frac{8}{25}(0) + \frac{5}{25}(0.32) + \frac{2}{25}(0.5) + \frac{10}{25}(0) = 0 + 0.064 + 0.040 + 0 = 0.104$$

---

## Question 1: What If TrainingHours Were a Daily Average?

> **What the question asks:** If the variable was hours per day (weekly ÷ 7) instead of
> hours per week, how would the tree change and what are the effects on predictions?

If TrainingHours is expressed as a daily average, each value is divided by 7:
$\text{TH}_{\text{daily}} = \text{TH}_{\text{weekly}} / 7$.

The coach's 8-hour threshold becomes $8/7 \approx 1.14$ hours per day.

**The split condition changes from** `TH_weekly >= 8` **to** `TH_daily >= 1.14`.

Since dividing by 7 is a positive monotonic transformation, the **relative ordering of all athletes
is preserved**. The same 12 athletes fall on the right (TH≥8 weekly = ≥1.14 daily) and the same 13
on the left. Every Gini computation and every prediction remains identical.

**Effect on predictions:** None. The tree is structurally and functionally identical; only the
numerical threshold printed in the flowchart changes (7.5 weekly → ~1.07 daily, using the
midpoint convention between the two nearest values).

> **Key principle:** Decision trees are invariant to monotonic transformations of continuous
> variables because they split by rank ordering, not absolute scale.

---

## Question 2: Classify Students A and B (Depth-2 Tree)

> **What the question asks:** Use the depth-2 tree to advise the coach. Show the path.

**Student A:** TH=9, FP=0, AG=1, PC=1

```
Step 1 — Root: TH <= 7.5?   → 9 <= 7.5?  NO  → go RIGHT
Step 2 — Node: FP <= 0.5?   → 0 <= 0.5?  YES → go LEFT (FP=0 leaf)
Leaf: n=2, Gini=0.5, Predict = Class 0
```
**Advice: Do NOT select Student A.**
(This leaf contains ID 10 [TT=1] and ID 20 [TT=0] — a perfect tie. The model cannot distinguish
this profile reliably at depth 2.)

**Student B:** TH=6, FP=1, AG=1, PC=0

```
Step 1 — Root: TH <= 7.5?      → 6 <= 7.5?  YES → go LEFT
Step 2 — Node: PC <= 0.5?      → 0 <= 0.5?  YES → go LEFT (PC=0 leaf)
Leaf: n=8, Gini=0.0, Predict = Class 0
```
**Advice: Do NOT select Student B.**
(This leaf is pure: all 8 athletes with TH<8 and no prior club are not in the top team.)

---

## Max-Depth Tree (max_depth=None)

> **What the question asks:**
> 1. How do you modify the code?
> 2. Build the full-depth tree by hand and discuss its characteristics.

### Code modification:

```python
# Depth-2 (original):
DecisionTreeClassifier(DATA, criterion='gini', splitter='best', max_depth=2)

# Full depth (remove max_depth or set to None):
DecisionTreeClassifier(DATA, criterion='gini', splitter='best')
# Equivalent: max_depth=None (the default)
```

Removing `max_depth` lets the tree keep splitting until every leaf is pure (Gini=0) or no further
split is possible.

---

### Building the Full Tree by Hand

The first two levels are identical (same root split TH<=7.5, same second splits FP and PC).
After depth 2, two leaves remain impure:

**Impure leaf A — Right branch, FP=0 (n=2):**

| ID | FP | AG | PC | TT |
|----|----|----|----|----|
| 10 | 0  | 1  | 1  | 1  |
| 20 | 0  | 0  | 1  | 0  |

Try **AttendanceGood:** AG=1 → {10} (TT=1) pure, AG=0 → {20} (TT=0) pure.
Weighted Gini = 0. ✓

Try **PriorClub:** both have PC=1 → no split possible (only one group).

**Best: AttendanceGood → resolves to 2 pure leaves of size 1.**

---

**Impure leaf B — Left branch, PC=1 (n=5):**

| ID | TH | FP | AG | TT |
|----|----|----|----|----|
| 5  | 7  | 1  | 1  | 1  |
| 8  | 5  | 0  | 1  | 0  |
| 13 | 2  | 0  | 0  | 0  |
| 14 | 6  | 1  | 0  | 0  |
| 25 | 6  | 0  | 1  | 0  |

Try **FitnessPass:** FP=1 → {5,14} (1 vs 1 → Gini=0.5), FP=0 → {8,13,25} (all TT=0 → Gini=0).
Weighted Gini = (2/5)(0.5) + (3/5)(0) = **0.20**

Try **AttendanceGood:** AG=1 → {5,8,25} (1 vs 2 → Gini=4/9≈0.44), AG=0 → {13,14} (all TT=0 → Gini=0).
Weighted Gini = (3/5)(4/9) + (2/5)(0) = **0.267**

Try **TrainingHours, threshold 6.5:** TH>6.5 → {5} (TT=1, pure), TH≤6.5 → {8,13,14,25} (all TT=0, pure).
Weighted Gini = 0. ✓

**Best: TrainingHours > 6.5 → resolves to 2 pure leaves.**

---

### Full Tree Flowchart

```
|--- TrainingHours <= 7.50               [Root:  n=25, Gini=0.4992]
|   |--- PriorClub <= 0.50               [Node:  n=13, Gini=0.1420]
|   |   |--- class: 0                    [Leaf:  n=8,  Gini=0.0]  IDs: 6,9,12,15,17,18,22,24
|   |   |--- TrainingHours <= 6.50       [Node:  n=5,  Gini=0.3200]
|   |   |   |--- class: 0               [Leaf:  n=4,  Gini=0.0]  IDs: 8,13,14,25 (all TT=0)
|   |   |   |--- class: 1               [Leaf:  n=1,  Gini=0.0]  ID: 5 (TT=1)
|--- TrainingHours >  7.50               [Node:  n=12, Gini=0.1528]
|   |--- FitnessPass <= 0.50             [Node:  n=12, Gini=0.1528]
|   |   |--- AttendanceGood <= 0.50      [Node:  n=2,  Gini=0.5000]
|   |   |   |--- class: 0               [Leaf:  n=1,  Gini=0.0]  ID: 20 (TT=0)
|   |   |   |--- class: 1               [Leaf:  n=1,  Gini=0.0]  ID: 10 (TT=1)
|   |   |--- class: 1                   [Leaf:  n=10, Gini=0.0]  IDs: 1,2,3,4,7,11,16,19,21,23
```

**As a diagram:**

```
                   [Root] n=25, Gini=0.4992
                 TrainingHours <= 7.5?
                /                        \
         YES (TH < 8)                NO (TH >= 8)
         n=13  Gini=0.1420           n=12  Gini=0.1528
               |                          |
        PriorClub = 0?              FitnessPass = 0?
        /           \               /              \
      YES             NO          YES               NO
  [Leaf] n=8    TH <= 6.5?    AG = 0?          [Leaf] n=10
  Gini=0         /       \     /     \           Gini=0
  Predict:0    YES        NO  YES      NO        Predict:1
           [Leaf]n=4  [Leaf]n=1 [Leaf]n=1 [Leaf]n=1
           Gini=0      Gini=0   Gini=0    Gini=0
           Pred:0      Pred:1   Pred:0    Pred:1
```

**Characteristics:**

| Property | Value |
|----------|-------|
| Max depth | 3 |
| Number of leaves | 6 |
| Leaves with Gini = 0 | 6/6 (all pure) |
| Overall weighted Gini | 0.0 (perfect fit on training data) |
| Samples per leaf | 8, 4, 1, 1, 1, 10 |

---

## Question 2 Revisited: Classify Students A and B (Full Tree)

**Student A:** TH=9, FP=0, AG=1, PC=1

```
Step 1 — Root: TH <= 7.5?       → 9 <= 7.5?  NO  → RIGHT (TH >= 8)
Step 2 — Node: FP <= 0.5?       → 0 <= 0.5?  YES → LEFT (FP = 0)
Step 3 — Node: AG <= 0.5?       → 1 <= 0.5?  NO  → RIGHT (AG = 1)
Leaf: n=1, Gini=0.0, Predict = Class 1
```
**Advice: SELECT Student A.** (ID 10 has the exact same profile and is in the Top Team.)

**Student B:** TH=6, FP=1, AG=1, PC=0

```
Step 1 — Root: TH <= 7.5?       → 6 <= 7.5?  YES → LEFT (TH < 8)
Step 2 — Node: PC <= 0.5?       → 0 <= 0.5?  YES → LEFT (PC = 0)
Leaf: n=8, Gini=0.0, Predict = Class 0
```
**Advice: Do NOT select Student B.** (Same result as depth-2 tree — this leaf is already pure.)

---

## Comparing the Two Trees

> **What the question asks:** Discuss the differences and why a coach might prefer a shallow tree
> even if it is not perfectly accurate.

| Property | Depth-2 Tree | Full-Depth Tree |
|----------|-------------|-----------------|
| Max depth | 2 | 3 |
| Leaves | 4 | 6 |
| Pure leaves | 2/4 | 6/6 (100%) |
| Training Gini | 0.104 | 0.000 |
| Rules per path | ≤ 2 conditions | ≤ 3 conditions |
| Risk of overfitting | Low | High |
| Prediction — Student A | Class 0 (wrong-ish: tied leaf) | Class 1 (correct match to ID 10) |
| Prediction — Student B | Class 0 | Class 0 |

**Discussion points:**

1. **Overfitting.** The full tree memorises the 25 training athletes perfectly. It has leaves
   containing just 1 observation each (e.g., only ID 5, only ID 10). These hyper-specific rules
   are unlikely to generalise — they may capture noise in this particular small sample rather than
   the true underlying pattern.

2. **Interpretability.** The depth-2 tree gives the coach a simple, memorable decision rule:
   *"Train at least 8 hours a week. If you do, you also need to have passed the fitness test."*
   This is two conditions at most — easy to explain to athletes and to apply consistently across
   seasons without a spreadsheet.

3. **Sample size.** With only 25 observations, deep splits (like isolating ID 5 by TH > 6.5)
   are based on very little evidence. A single atypical athlete (ID 5: 7 hrs, selected) should not
   drive a new branch of the tree; the coach can handle edge cases as exceptions.

4. **Bias-variance trade-off.** A shallower tree has higher bias (more training errors) but lower
   variance (more stable across different samples). With small datasets, lower variance is
   statistically preferable. A shallow tree captures the dominant patterns; a deep tree also
   captures the noise.

5. **Practical transparency.** Coaches and athletes need to trust and understand selection
   criteria. A tree so deep that it selects one specific athlete profile is not actionable advice —
   it is opaque and looks arbitrary. A shallow tree functions as a transparent policy.

> **Key point:** A shallow tree may make a few mistakes on training data, but it captures the
> general patterns that are more likely to hold for future athletes. Perfect accuracy on 25
> historical athletes is not the goal; generalising to future recruits is.
