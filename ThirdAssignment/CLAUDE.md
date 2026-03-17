### You will only deal with Question 2 and 3 (unless explicitly told otherwise). Produce one question at a time.


### Virtual environment

Please note I am using a virtual environment. ( @.venv ), it's a folder in the @ST239_Assignments folder, you have to run pip install in the terminal with that environment active in order to import libraries. Note that the main ones are already there (matplotlib, pandas, sklearns, stats, etc.) If you run into trouble, ask the user to manually pip install the libraries into the venv. 

### Q3 Method choices (confirmed by user):
- Logistic Regression + Random Forest
- Cross-validation for both, compare with suitable metrics (accuracy, F1, AUC etc.)

For each Q2 and Q3, you have both a python file and a markdown report file. (for Q2, @q2workspace.py and @q2_report.md; for Q3, @q3workspace.py and @q3_report.md)

As for code production and comments follow the coding style in any of the jupyter notebook files in the Past_Examples folder. 

Read @MD_assignmentInstructions.md for more details on the assignment structure and requirements. 





### Report structure: 

all relevant methodological choices, reasoning, and steps in the analysis
are clearly explained in the report. The report should be written so
that a reader can fully understand the motivation, approach, and key
components of the analysis without needing to inspect the code.
Here is a suggestion on how to structure the report and its parts:

Introduction: describe the dataset and the observed variables, providing
context for the analysis.

ExploratoryDataAnalysis: A short exploratory data analysis to present
the main aspects of the dataset exploration, using summary statistics and
visualizations (univariate, pairwise). At this stage, you should translate
preliminary findings into language that is accessible to a general, non-
expert audience. End this section by clearly stating the aims of the tech-
nical analysis.

Technicalpart: Carry out appropriate statistical analysis by applying sta-
tistical and machine learning techniques presented in the module (more
details under the relevant questions).

Conclusions: Interpret the results from both expert and non-expert perspectives.
Discuss potential extensions or limitations of the analysis.
In the report, aim to communicate non-technical insights in a way that
would be accessible to an audience that has not attended the module but
should be able to understand and make decisions based on the analysis
findings. The technical part should give details of the analysis undertaken
and the main ideas of the quantitative tools that were presented in the
lectures.

Style suggestions
– Plot References: You are encouraged to add figures to the report if they
are helpful and relevant for your narrative. Every figure included in the
report has to be produced by your Python code, you should clearly refer
to the specific part of code that you used to generate it.

