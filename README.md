# eThekwini 2026 Local Election ML Analysis

## 1. Project Overview

This project develops an evidence-based Machine Learning analytics solution for the **2026 South African Local Government Elections** in **eThekwini Metropolitan Municipality**.

The purpose of the project is not to provide a political opinion. The objective is to demonstrate how historical election data can be transformed into a technically defensible predictive solution for:

* Projected aggregate party vote share and vote totals
* Projected leaders in three selected wards
* Projected voter turnout
* Analysis of whether a single relevant party reaches a projected majority
* Model validation and evaluation

The analysis uses information available at the time of the examination and produces **model-derived estimates for 4 November 2026**.

---

## 2. Data Sources

The main datasets are official election results for eThekwini Metropolitan Municipality obtained from the Independent Electoral Commission (IEC) of South Africa.

### 2016 Municipal Election Data

Independent Electoral Commission (IEC):

https://www.elections.org.za/content/Elections/Downloadable-results/Detailed-results-data--2016-Municipal-Elections/

### 2021 Municipal Election Data

Independent Electoral Commission (IEC):

https://www.elections.org.za/pw/Elections-and-results/Municipal-Elections-2021

### 2021 IEC Results Dashboard

https://results.elections.org.za/dashboards/lge/leaderboard.aspx/1000

The project files contain the eThekwini ward-level datasets used in the analysis.

---

## 3. Dataset Understanding

The 2016 dataset contains ward and voting-district election observations, including:

* Municipality
* Ward
* Voting District
* Voting Station
* Party
* Ballot Type
* Registered Voters
* Spoilt Votes
* Total Valid Votes
* Election date information

The 2021 dataset contains equivalent election-result information.

The datasets were inspected for:

* Missing values
* Duplicate observations
* Invalid dates
* Negative vote/electorate values
* Vote totals exceeding registered voters
* Ward and voting-district structure
* Party coverage
* Ballot types

Only **Ward ballot** records were used for the ward-level modelling.

---

## 4. Data Preprocessing

The preprocessing workflow included:

1. Loading the 2016 and 2021 CSV datasets.
2. Cleaning column names and text fields.
3. Converting date fields to datetime format.
4. Checking for missing values and duplicates.
5. Checking for invalid numerical values.
6. Filtering the datasets to Ward ballot observations.
7. Aggregating voting-district observations to ward level.
8. Calculating party vote shares for each ward.
9. Calculating ward turnout using:

   **Turnout = (Valid Votes + Spoilt Votes) / Registered Voters**

5 relevant party categories were modelled:

* ANC
* DA
* EFF
* INDEPENDENT
* IFP

Other parties were not incorrectly redistributed into these categories.

---

## 5. Geographic Alignment

A major consideration was that ward identifiers cannot automatically be assumed to represent exactly the same electorate across different election years.

The 2016 data contained **110 wards**, while the 2021 data contained **111 wards**.

Voting-district composition was therefore compared between the two elections.

There were:

* 110 common ward identifiers
* 53 wards with identical voting-district composition

These **53 geographically stable wards** were used for the 2016 → 2021 ward-level modelling transition.

This approach reduces the risk of comparing different physical electorates simply because they have the same ward identifier.

---

## 6. Exploratory Data Analysis

Historical analysis examined:

* Party vote-share changes between 2016 and 2021
* Ward-level party relationships
* Turnout changes
* Registered voter changes
* Relationships between historical party performance and subsequent performance
* Geographic stability of wards

The analysis showed substantial changes between the 2016 and 2021 elections, including changes in the relative performance of major parties and a substantial decline in ward-level turnout.

These historical patterns were used as inputs to the predictive modelling stage.

---

## 7. Feature Engineering

The main modelling features were:

* 2016 ANC vote share
* 2016 DA vote share
* 2016 EFF vote share
* 2016 Independent vote share
* 2016 IFP vote share
* 2016 turnout
* 2016 registered voters

The target variables were the corresponding 2021 party vote shares and 2021 turnout.

Historical change variables were investigated during feature engineering but were not used as predictors where they could introduce information leakage.

---

## 8. Machine Learning Model

### Linear Regression

Linear Regression was selected because:

* The prediction targets are continuous values.
* Vote share and turnout are numerical targets.
* The comparable ward sample is relatively small.
* The model is interpretable.
* It is computationally efficient and reproducible.
* It provides a technically appropriate baseline for the available data.

A StandardScaler was used before Linear Regression through a Scikit-learn Pipeline.

Separate models were developed for:

* ANC vote share
* DA vote share
* EFF vote share
* Independent vote share
* IFP vote share
* Turnout

---

## 9. Validation and Evaluation

Because only 53 geographically stable wards were available, **5-fold cross-validation** was used.

The evaluation metrics were:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R²

A persistence baseline using historical values was also considered during the modelling process.

Performance varied between targets. The model performed more strongly for some parties than others, while the Independent target had weaker predictive performance.

The evaluation results are displayed in the Streamlit dashboard.

---

## 10. 2026 Projection

The 2021 election was treated as the latest available historical state for generating the 2026 projection.

The trained models were applied to the available 2021 ward-level information.

Predictions were bounded between 0 and 1 because Linear Regression does not inherently enforce valid vote-share or turnout limits.

The selected party predictions were not artificially normalised to 100%, because other parties remain relevant.

---

## 11. Selected Wards

Three wards were selected for ward-level analysis:

### Ward 59500001

A historically ANC-dominant stable ward.

### Ward 59500036

A historically DA-dominant stable ward.

### Ward 59500090

A historically competitive/switching stable ward.

The selected wards were chosen because they are geographically stable and provide different historical voting patterns for analytical comparison.

---

## 12. Streamlit Dashboard

The Streamlit application provides the following sections:

### Overview

Summarises the project and major projected outputs.

### Historical Analysis

Shows historical party performance and turnout patterns.

### 2026 Model Outputs

Displays projected aggregate party vote shares, projected vote totals and turnout estimates.

### Selected Wards

Displays projected party shares, projected ward leaders and turnout for the three selected wards.

### Model Evaluation

Displays MAE, RMSE and R² results from cross-validation.

### Uncertainty & Limitations

Explains geographic, temporal and modelling limitations.

---

## 13. Interpretation of Results

All 2026 values must be interpreted as **analytical model outputs**, not guaranteed election outcomes.

The project distinguishes between:

1. Historical observations from the 2016 and 2021 elections.
2. Variables and estimates derived from historical data.
3. Predictions generated for the 4 November 2026 election.

The model does not predict political opinions, campaign outcomes or which parties will negotiate a governing coalition.

If no single relevant party reaches a projected majority, the dashboard indicates that coalition formation may be required, but it does **not** predict which parties would form such a coalition.

---

## 14. Limitations

The main limitations are:

* Only two municipal election years were available for this transition.
* Only 53 wards had identical voting-district composition between 2016 and 2021.
* Ward boundaries and voting-district structures can change between elections.
* Election outcomes can be affected by factors not represented in historical vote data.
* Linear Regression does not inherently enforce probability/share constraints.
* Predictive performance differs between parties.
* The Independent model has weaker validation performance.
* The 2026 values are estimates and should not be interpreted as certain outcomes.

---

## 15. Project Files

The project contains:

* `app.py` — Streamlit dashboard and predictive analysis
* `eThekwini_2026_Local_Election_ML_Analysis.ipynb` — reproducible analysis notebook
* `ETH_Ward (2016).csv` — 2016 eThekwini ward election data
* `ETH_Ward (2021).csv` — 2021 eThekwini ward election data

---

## 16. Technology Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Jupyter Notebook

---

## 17. Reproducibility

To run the Streamlit application, place the following files in the same directory:

```text
app.py
ETH_Ward (2016).csv
ETH_Ward (2021).csv
```

Then run:

```text
streamlit run app.py
```

The application loads the election datasets, performs preprocessing, geographic alignment, feature preparation, model evaluation and prediction automatically.

---

## 18. Final Note

This project demonstrates an end-to-end Machine Learning Life Cycle:

**Data Acquisition → Data Understanding → Preprocessing → EDA → Feature Engineering → Modelling → Validation → Evaluation → Prediction → Streamlit Deployment**

The emphasis is on technical defensibility, reproducibility, appropriate validation and transparent interpretation rather than certainty about the 2026 election outcome.
 ## 15. Development Errors, Challenges and Improvements

During development, several technical issues were encountered and corrected as part of the Machine Learning Life Cycle.

### 15.1 Dataset File and Loading Errors

The application initially referenced incorrect CSV filenames. The filenames were checked against the actual project directory and corrected so that the application loads the intended 2016 and 2021 datasets.

### 15.2 Incorrect Dataset Version

An early version of the 2021 dataset was incorrectly identified during development. This was detected by comparing the datasets and checking their dimensions and election characteristics. The correct 2021 dataset was then used.

### 15.3 Ballot Type Issue

The 2016 dataset contained both Ward and PR ballot observations, while the 2021 dataset used for this analysis contained Ward observations. Including both ballot types would have mixed different electoral measures. The analysis therefore explicitly filtered the data to Ward ballot observations for ward-level comparison.

### 15.4 Geographic Alignment Issue

A major analytical issue was discovered when comparing ward identifiers between elections. The same ward identifier does not necessarily guarantee the same geographic electorate because voting-district compositions can change.

Instead of assuming that all matching ward numbers were directly comparable, voting-district composition was compared between 2016 and 2021. Only wards with identical voting-district composition were retained for the historical transition model, resulting in 53 geographically stable wards.

### 15.5 Missing Party-Ward Combinations

Some wards did not contain a row for every selected party. This does not necessarily indicate a missing ward. It can mean that the party received no recorded votes in that ward.

The modelling table was therefore aligned by ward and missing selected-party combinations were represented as zero rather than causing the modelling process to fail.

### 15.6 Model Prediction Constraints

Linear Regression can produce predictions below 0 or above 1 when predicting vote shares or turnout. Since these values represent proportions, predictions were bounded to the valid 0–1 range before interpretation.

### 15.7 Model Performance Differences

The model did not perform equally well for every target. In particular, the Independent target produced weaker validation performance than several other targets.

This was retained and documented rather than hiding the weaker result. Cross-validation metrics are therefore included in the dashboard so that the reliability of different predictions can be assessed.

### 15.8 Dashboard Development Issues

The Streamlit application initially experienced dependency and file-path errors during development. These were resolved by removing an unnecessary plotting dependency from the application and correcting the dataset file references.

These errors were treated as part of the development and debugging process rather than changing the analytical results to obtain a desired outcome.

### 15.9 Important Analytical Lesson

The main lesson from the development process was that a working model is not sufficient on its own. Data quality, geographic comparability, leakage prevention, validation and transparent documentation are necessary before model predictions can be interpreted.
