# ST239: Assignment 2

## Instructions

### The assignment is composed of 3 Questions worth a total of 60 points, respectively,


## Q1. Bagging vs Boosting for credit fault prediction

The dataset credit_default.csv collects information on 6,000 customers from a fictitious
bank regarding their financial health; this inclused the variable default which records
whether the customer has experiences a credit default (1) or not (0).

You will consider ensemble models of decision trees - a random forest (RF) and a boosting
one (with XGBoost)- to compare their use for making predictions on future customers and
their potential of credit default.

- Hyperparameter tuning First consider each ensemble model separately and
    tune the model parameters via cross-validation, in particular:
       - Create a CV grid for the random forest exploring hyperparameters such:
          max_depth (2, 3, 5, None) , max_features (choose a few suitable values
          according to the problem) and number of trees (50, 200, 500, 1000) and
          minimum samples leaf (1, 5, 10).
       - Create a CV grid for XGBoost exploring hyperparameters such: learning
          rate (0.01, 0.03, 0.05, 0.1), max_depth (2, 3, 4, None), subsample (0.7, 1)
          and number of trees ((50, 200, 500, 1000)).

```
Use the logloss as splitting criterion for both, use the whole dataset for cross
validation with a 5-fold. Compare each ensemble over the selected tuning pa-
rameters in terms of accuracy and roc_auc and explain their meaning.
```
- For each ensemble model type, summarise your findings. Below are a few ques-
    tions that should guide your reflection and your final choice for the two respective
    ensembles:
       - How does random forest accuracy and auc vary as the number of estimators
          increases? How about the minimum sample leaf and max_depth? You
          may use a plot to show the scores and the selected hyperparameter or a
          table to report your findings and use it for the discussion.
       - What about overfitting? Does the accuracy of the training test decrease
          with respect to the training one? Is it the case for both RF and XGBoost
          or do they show a different behaviour?

```
HINT: you may focus on a few parameters at the time if the computing time
is large, it would also make it easier for interpretation as you can let one hy-
perparameter vary and keep the others fixed. You are free to investigate other
hyperparameter values that you rate as important.
```

Model Comparison

- State/indicate the hyperparameter combination returning the two best models,
    in terms of accuracy, respectively for each approach (RF and XGB). Would
    your selection chance if you were to look at the best model in terms of log-loss
    or auc?
- Considering the two best models found above, extract the top 5 most impor-
    tant features, respectively.
       - Do the models agree on the top 5 predictors?
       - Looking at the overall importance ranking, is variables importance consis-
          tent across models?
       - Is the variable importance equally spread or highly concentrated on few
          ones?
       - Are the important predictors plausible in a credit-risk context?

Final evaluation. In the first part of the question, we used cross-validation to tune
model hyperparameters and found the ’best models’ in terms of log-loss, accuracy
and/or auc. Finally, you will use repeated train/test splits (30 runs) to validate the
final model.

- Use a train/test split (usual 80/20) 30 times with different seeds.
- For each run, fit the best tuned RF and best tuned XGB (in the log-loss sense)
    and compute: Accuracy and Recall.
- For each of the scores above, compare the two models by plotting error bars
    hihgligting the mean and standard deviation across runs.
- Present your latest findings:
    - Which model has the higher average Accuracy, Recall across different runs?
       Provide a recommendation on the best model based on your finding.
    - Which model has lower variability across runs/seeds?
    - Which metric would you prioritise in credit default and why?

```
[15 marks.]
```

## Q2. ‘Planting’ your decision tree

A university coach wants to come up with a criterion to decide whether a student should be
selected for the top BUCS Rugby Team.

The coach has a dataset collecting information on university student athletes (the ob-
served statistical units) from the year before on the following variables:

- TrainingHours: training hours per week (numeric)
- FitnessPass: 1 if passed a fitness test, 0 otherwise
- AttendanceGood: 1 if good attendance, 0 otherwise
- PriorClub: 1 if previously played for a club, 0 otherwise

```
the dataset is relatively small and provided below:
```
```
ID TrainingHours FitnessPass AttendanceGood PriorClub TopTeam
1 10 1 1 1 1
2 9 1 1 0 1
3 8 1 0 1 1
4 11 1 1 1 1
5 7 1 1 1 1
6 6 1 1 0 0
7 12 1 1 0 1
8 5 0 1 1 0
9 4 0 1 0 0
10 8 0 1 1 1
11 9 1 0 0 1
12 3 0 0 0 0
13 2 0 0 1 0
14 6 1 0 1 0
15 7 0 1 0 0
16 10 1 0 1 1
17 5 1 1 0 0
18 4 1 0 0 0
19 8 1 1 0 1
20 9 0 0 1 0
21 11 1 1 1 1
22 3 1 1 0 0
23 12 1 0 1 1
24 7 1 0 0 0
25 6 0 1 1 0
```

The coach summons youin their office and asks you to help them by creating a decision
tree for the task. Sadly, you don’t have your laptop with you and the coach’s desktop does
not have Python installed. You will have to do this by hand! Below is a suggested workflow:

- Consider the Gini impurity as splitting criterion. Write down the computation
    for the Gini impurity at root node (using all 25 observations).
- If you had your laptop, your would be using the following code:
    DecisionTreeClassifier(DATA, criterion=‘gini’, splitter=‘best’,
    max_depth=2)
    Explain what would be the output of this code in terms of the decision tree
    properties (diagram flowchart).
- The coach wants to make sure that their athletes are going to meet the required
    hours of training: 4 sessions of 2 hours per week. Thus, you should force your
    first split to use this variable for the tree construction. Show how the athletes
    IDs would be split into the two groups and compute the Gini impurity index
    for each child node resulting in the split and compute the combined/weighted
    Gini impurity. Show the impurity reduction from this split.
- Following the code and the split above, choose the variable that would give
    the best-second level split based on the impurity index by computing all three
    different scenarios and justify your final choice.
- Write down the flow diagram representing the final tree: consider the root node
    and report the starting impurity index, show the splits based on the first variable
    and the first-level intermediate nodes, the Gini impurity of each leaf and the
    final split, as if it was a Python output.
- Question 1: Imagine that instead of TrainingHours per week, you were pro-
    vided with a daily avarage (over 7 days). Describe how your decision tree would
    change and what would be the effects in terms of predictions.


- Question 2: Using the tree that you just developed, consider the following
    students and advice the coach on whom to select for the 1st team:
       - Student A: TrainingHours = 9, FitnessPass = 0, AttendanceGood = 1,
          PriorClub = 1
       - Student B: TrainingHours = 6, FitnessPass = 1, AttendanceGood = 1,
          PriorClub = 0

```
Show the decision making process i.e. the path through the decision tree.
```
- How would you modify the code above if you wanted to consider all variables
    and reach the maximum depth possible? Build this second decision tree, by hand
    and discuss its characteristics e.g. final combined purity, number of samples in
    each leaf and write down the flowchart.
- Answer question 2 using your newly developed decision tree.
- Comment on the differences between the two decision trees. Why might a
    shallow tree be preferred by a coach, even if it is not perfectly accurate?

```
[10 marks.]
```

## Q3. PCA in practise

The dataset Q4_workers.csv records information about the job of 1,200 individuals. A
description of the observed variables is provided in the table below:

```
Variable Type Description
age Integer Age of the worker (years).
annual_income_gbp Integer Annual gross income in GBP.
standing_hours_per_day Continuous Average number of hours per workday spent standing.
lifting_hours_per_day Continuous Average number of hours per workday spent lifting or carrying loads.
manual_intensity_score Integer (1–10) Self-reported intensity of manual labour (1 = very low, 10 = very high).
repetitive_motion_score Integer (1–10) Self-reported frequency of repetitive movements at work.
seated_hours_per_day Continuous Average number of hours per workday spent seated.
computer_hours_per_day Continuous Average number of hours per workday spent using a computer.
meetings_hours_per_week Continuous Average number of hours per week spent in meetings.
chronic_pain Binary (0/1) Indicator equal to 1 if the worker reports chronic musculoskeletal pain, 0 otherwise.
```
### Table 1: Description of variables in the ‘workers’ dataset.

In this exercise, you will explore the structure of the explanatory variables (excluding the
target chronic_pain) using Principal Component Analysis (PCA). Later, you will fit
a logistic regression model to predict the probability of reporting chronic pain using the
original variables and the reduced Principal Components.

```
Principal Components Analysis - Suggested steps
```
- Data Inspection Start by exploring the dataset with some
    numerical summary statistics and discuss what are the main characteris-
    tics of the dataset e.g. scale differences between the variables, correlation
    structure between variables etc. It can also help to produce some visualisations
    but these are optional.
- Perform PCA and explain whether you should perform data processing i.e.
    mean-centering and/or standardisation and why.
- Use the outputs available from PCA (eigenvalues, eigenvectors etc) to interpret
    the results:
       - Can you ‘give a name’ to the Principal Components using the loadings to
          provide an interpretation of the ‘new‘ variables? Plot the factor loadings
          for the 1st-and-2nd PC, 1st-and-3rd PC and 2nd-and-3rd PC to aid your
          analysis.


- Project the data into a lower dimensional space using the first two and three
    Principal Components respectively to create a 2d and a 3d scatterplot. Is there
    anything that you can observe? Are there any visible clusters? Does PC
    provide an additional meaningful separation? Consider how PC variables relate
    directionally to your plot.
- Discuss the dimensionality reduction criteria that can be adopted in PCA and
    apply them to your analysis (use numerical results and plots to present your
    findings). How many components would you retain for this dataset based on
    them? (If not all approaches agree on the number of PC to retain, give a
    justification for your choice)

```
[12 marks.]
```
Logistic Regression - Suggested steps Perform a standard logistic regression
analysis using chronic_pain as target.
It is not required to perform diagnostics.
Note: evaluating the predictive performance of the model using train/test split or
cross validation is optional but bonus marks (up to 5) may be awarded.

- Data Inspection Additionally to what you have done in the first part of the
    exercise, consider the target variable chronic_pain in terms of its summary
    statistics and its correlation to the original variables as well as with the variables
    retained after performing PCA (Consider only the projected data for the PCs
    you choose to retain, not the full dimensional rotation).
- Fit logistic regression with original variables
    - Select relevant variables on the basis of significance, report and evaluate
       the coefficient estimates.
    - Which predictors appear most strongly associate with chronic pain?
    - Are some variables difficult to interpret jointly?


- Fit logistic regression with PCA variables
    - Fit a logistic regression using the selected principal components.
    - Are the PCs predictors associated with the risk of chronic pain? Interpret
       the coefficients and the odds for the model and discuss your findings.
- Compare the two final models in terms of: number of parameters used, inter-
    pretability, fitting performances (AIC, pseudo-R^2 etc).

[8 marks.]
The following questions only require a textual answer and reflection on the analysis
you just carried out.

- Question 1: PCA was performed without using the target variable. Why can
    principal components still be predictive of chronic pain?
- Question 2: Is PCA guaranteed to improve predictive performance? Why or
    why not? (Think in terms of accuracy and goodness of fit)
- Question 3: What is the main conceptual difference between Feature selection
    and Feature extraction (here PCA)?

```
[5 marks.]
```

