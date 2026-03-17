

ST239:  Assignment 3
## Instructions
This assignment is worth 50% of the total module mark.
The assignment is composed of 3 Questions worth a total of 100 points.
Submission Components  Your submission should include the following docu-
ments:
- One pdf with your attempt of Question 1. You can do it with pen and paper
and scan the page as a pdf (please, use a clear and readable handwriting) or
you can use any electronic device and software (powerpoint, word, Canva, any
drawing apps for tablets) and save your work as a pdf.
- Two Python Notebooks (with .ipynb extension) containing the code used
to develop the reports for Questions 2 and 3, respectively.
The code should be fully reproducible (i.e. I should be able to run it in my local
machine with no errors) and well-commented.
Style suggestions
– For each chunk of code or procedure used, you should add a small comment
in code or a description on its functionality and how it is being used in a
text cell.
– Written comments, i.e. Markdown text cells, in the Notebooks are not go-
ing to be considered for evaluation of the report and will only be counted
as comments to aid the analysis.
Please, avoid lengthy or repetitive text in your code; you should
add all relevant information, comments and findings in the re-
port.
– DO NOT submit the datasets. I will place your code in a folder with them,
so you should access them with a simple path command, e.g. file-name.csv


- Two reports (text file such as pdf, word..)  containing the analysis of the
dataset in Question 2 and 3, respectively, with both technical and non-technical
sections.
The report is the primary component used to evaluate your anal-
ysis. The accompanying code, which supports and implements the
analysis, will also be reviewed to ensure it is consistent with the
narrative presented in the report. Students should ensure that all
relevant methodological choices, reasoning, and steps in the analysis
are clearly explained in the report. The report should be written so
that a reader can fully understand the motivation, approach, and key
components of the analysis without needing to inspect the code.
Here is a suggestion on how to structure the report and its parts:
## -
Introduction: describe the dataset and the observed variables, providing
context for the analysis.
## -
ExploratoryDataAnalysis: A short exploratory data analysis to present
the main aspects of the dataset exploration, using summary statistics and
visualizations (univariate, pairwise). At this stage, you should translate
preliminary findings into language that is accessible to a general, non-
expert audience. End this section by clearly stating the aims of the tech-
nical analysis.
## -
Technicalpart: Carry out appropriate statistical analysis by applying sta-
tistical and machine learning techniques presented in the module (more
details under the relevant questions).
## -
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
to the specific part of code that you used to generate it. You do not have


to include all plots in the report but only the ones that you believe are
relevant. You can also refer to a plot that you produced and has been useful
to gather insight on the dataset without putting it in the report.
– Report length: approximately 1500 words (not strict)



Question 1: A MindMap of ST239
Create a MindMap that visually organises the various Statistical and ML topics
that were presented in the module. If you are unsure on how to build one, below
is an example of a MindMap organising sports based on main categories/sub-
categories, their similarities, differences and peculiarities (it is not the only way
to do it):
CamScanner
Figure 1: MindMap of Sports. Note that I choose to first divide sports depending
on the equipment being used (3 main categories); for some categories I then consider
the specific “style” of the sport (e.g. combat or individual performance score; ball or
ball/raquet), finally each leaf/sport has a small squared tag with some specific info
and also a small branching mentioning possible variants (I could have also listed them
inside the squared tag). I did not provide a description to all sports simply for the
sake of compactness of the example. You should include a brief description to
all menthods/leaves included.
MindMaps can be a useful tool for study and revision of entire modules and
help acquiring new concepts by framing them in the right context. The goal


of this question is to help you synthetise knowledge and develop a structural
understanding of the methods and their connections.
You have a lot of freedom on how to organise your MindMap, below are some
general guidelines:
– Central node: place the main topic or idea (here, the module)
## – Branches:
∗ Consider the key themes which would create the major categories and
subcategories (more hints below).
∗ Use keywords or short phrases to explain the branching (some branch-
ing may be symmetrical and appear in two different sides of the tree)/
∗ Avoid overbranching: try to limit each branch to a few sub-branches,
and avoid overly detailed breakdowns at the initial stages.
– Leaves: ideally your final leaves should represent a method/technique of
the module. Give a short description of the key idea (i.e. one important
equation; a few sentences that would frame the method; its complexity
and scalability, if relevant, etc..).
– You can provide a small caption (similar to the one above) to give a general
description of the organisation of your MindMap.
Here are some relevant (but not exhaustive) aspects to consider when mapping
the module topics i.e. the Data Science methods:
(i) what is the communal goal of the methods in the same (sub)category?
(ii) what are the characteristics of the datasets being used?
(iii) in model building and parameter estimation, what are the assumptions
and how are the parameters found? Is the model probabilistic?
(iv) what is the level of interpretability (white, black, grey box)?
[5 marks.]


## Question 2: Exploring Urban Profiles Using Dimensionality
Reduction and Clustering
The dataset Q2_cities.csv collects information on a set of fictitious cities.
Each row represents a city, while the columns contain numerical indicators
related to demographics, economic performance, infrastructure, environmental
characteristics, and sectoral composition; more information can be found in the
following table.
VariableUnit / Type  Description
populationcountTotal population of the city.
area_km2km
## 2
Total land area of the city.
density_per_km2people/km
## 2
Population density.
gdp_per_capita_usdUSDGross domestic product per
capita.
avg_salary_usdUSDAverage  annual  salary  of
workers.
unemployment_rate_pct%Unemployment  rate  of  the
labour force.
median_ageyearsMedian age of residents.
bachelors_or_higher_pct%Share of adults with at least a
bachelor’s degree.
diversity_index_0_100index (0-100)   Cultural and demographic di-
versity indicator.
transit_score_0_100index (0-100)   Public transport accessibility
score.
housing_cost_indexindexRelative housing price indica-
tor.
green_space_m2_per_capitam
## 2
/personGreen space available per res-
ident.
air_quality_indexindexAir pollution indicator.
tech_employment_pct%Workforce employed in tech-
nology sectors.
manufacturing_employment_pct  %Workforce employed in manu-
facturing sectors.
tourism_index_0_100index (0-100)   Tourism activity indicator.
innovation_index_0_100index (0-100)   Innovation and research activ-
ity indicator.
startup_density_per_100kper 100kNumber  of  startups  per
100,000 residents.
renewable_energy_pct%Share of energy from renew-
able sources.
commute_time_minminutesAverage commuting time.
Table 1: Variables included in the fictitious cities dataset.
The goal of this exercise is to explore whether cities can be grouped based on


these characteristics.  To do this, you will combine dimensionality reduction
techniques and unsupervised clustering methods.
Below are some guidelines on how to organise your analysis; you should then
present your exploration of the dataset, the rationale on how and why you chose
to use certain techniques and your findings in the light of what you have learnt
during the module in the form of a report.
– Begin by exploring the dataset: inspect the variables and their
nature, consider how they relate to one another and reflect on their
meaning to inform the analysis.
– Before applying clustering methods, consider whether the data re-
quire preprocessing. Discuss and justify any steps you choose to
perform before applying more advanced methods.
– The dataset contains many variables, which can make it difficult
to visualize and interpret structure in the data. You may consider
dimensionality reduction with the objective of keeping a set of
features for visualisation as well as to summarise the multiple vari-
ables into latent components. You should dedicate a section of the
report to describe how you perform this important step, the princi-
ples of the methods, how you apply it and the output.
– Visualizing the cities in the space defined by the reduced spece may
reveal patterns or possible groupings. The final and crucial compo-
nent of this analysis is to attempt to identify groups of cities with
similar characteristics. You may consider different clustering ap-
proaches, such as:
∗ hierarchical clustering
∗ k-means clustering
∗ density-based clustering


These methods rely on different assumptions about the struc-
ture of the data, and they may produce different results. It is
important you present your trials under each approach, keeping
in mind the main differences in these techniques and the ratio-
nale, which explains the generated clustering. Possible points
to reflect on include:
∗ How many clusters appear to be present in the data
∗ Whether clustering should be performed in the original feature
space or in the reduced one.
∗ How the results differ across clustering methods and whether
you can spot any anomalies.
Once clusters have been identified, you may attempt to interpret
them based on the most defining features of each clusters. Do clus-
ters correspond to plausible urban profiles (e.g. dense metropolitan
centers, environmentally oriented cities, rural or industrial regions
etc)?
– Finally, reflect on the analysis process:
∗ How sensitive are the clustering results to preprocessing
choices?
∗ Do different clustering algorithms produce similar groupings?
∗ Discuss whether the clusters represent meaningful patterns in
the data or reflect some design assumptions.
[50 marks.]


Question 3: Analysis of Online Shopping Dataset
The dataset Q3_OnlineShopping.csv contains information about online shop-
ping sessions made by anonymised online shoppers recording a series of nu-
merical and categorical variables. The dataset was formed so that each session
would belong to a different user in a 1-year period. A description of the observed
variables is provided below:
VariableTypeDescription
devicecategorical  Device used for the session (mobile, desk-
top, tablet).
visitor_typecategorical  Indicates whether the visitor is new or re-
turning.
time_of_daycategorical  Time period of the session (morning, af-
ternoon, evening, night).
day_typecategorical  Indicates whether the session occurred on
a weekday or weekend.
session_duration_min    numericDuration of the browsing session in min-
utes.
page_viewsnumericTotal number of pages viewed during the
session.
items_to_cartnumericNumber of items added to the shopping
cart during the session.
days_since_last_visitnumericNumber of days since the visitor’s previous
visit.
main_product_category  categorical  Main product category viewed during the
session.
delivery_speedcategorical  Delivery option offered for the product
(standard, express, next-day).
return_policycategorical  Type of return policy available (basic or
extended).
product_rating_mean    numericAverage rating of the product viewed (1-5
scale).
purchasebinaryTarget variable indicating whether a pur-
chase occurred during the session.
Table 2: Variables in the online shopping session dataset.


## Analysis Guidelines
– The goal of this study is to use the variable “purchase” as target.
– As outlined in the report instructions, your analysis should include
an introduction and exploration of the essential characteristics of the
dataset using summary statistics and visualisation tools.
– Discuss any data pre-processing; e.g. dropping variables or scaling
variables, one-hot-encoding and trasformation of boolean into nu-
merical for any specific purposes.
– The technical part is the heart of the analysis: perform different
modelling approaches suitable for this dataset among the ones that
have been presented in the lectures and labs. The number and choice
of methods used will improve the depth of your analysis and make
it more complete. Consider at least two methods, test using cross-
validation, and compare their performances using suitable metrics.
Each method should be accompanied by a general description and
motivation of its use in extracting insights and addressing the re-
search question. For each approach, where it applies: recall the un-
derlying assumptions, consider the choice of variable selection and
of tuning parameters. Interpret the outcome of the method using
the relevant outputs (e.g. summary plots, variable significance, per-
formance metrics). This discussion should be suitable for an expert
knowledge that is familiar with the technicalities of the methods.
– In your discussion, include a non-technical interpretation of the
methods and their outputs that is suitable for a non-expert audi-
ence.
– Conclude with a summary of the outcomes of your analysis: compare
methods using suitable metrics and benchmarks; discuss potential
limitations and differences between the methods, suggest if there
could be potential extensions.
[45 marks.]