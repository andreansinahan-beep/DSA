# Loan Approval Prediction System

CC 204: Data Structures and Algorithms - West Visayas State University, CICT

## Description
Classifies a loan applicant as **Approved (Y)** or **Rejected (N)** using three
algorithms written from scratch (no machine learning library):

| Algorithm | Data structure |
|---|---|
| K-Nearest Neighbors (k = 7) | list + min-heap (`heapq`) |
| Decision Tree (Gini, max depth 5, min size 10) | binary tree of `Node` objects |
| Naive Bayes | hash maps (`dict`, `Counter`) |

## Folder structure
```
loan-approval-dsa/
  app.py             Streamlit web app
  algorithms.py      data loading, preprocessing, algorithms, benchmark
  data/train.csv     dataset (download it yourself)
  results/           saved benchmark CSV files
  requirements.txt
  README.md
```

## Dataset
Download `train.csv` (614 rows) from Kaggle's **Loan Prediction Problem Dataset**
and save it as `data/train.csv`.

An optional `data/sample.csv` enables the prediction-page demo buttons. Applicant
CSV files and generated results are excluded from Git; add only datasets you are
authorized to share.

## Install
Run these inside the `loan-approval-dsa` folder:
```
pip install -r requirements.txt
```

## Run the web app
```
streamlit run app.py
```

## Run from the command line
```
python algorithms.py data/train.csv
```
This prints the preprocessing summary, per-trial and summary results, confusion
matrix counts, and the Decision Tree's depth, node count and root feature.

## Design choices (be ready to explain these)
- `Credit_History` is converted to float but treated as a category (0 or 1) by the
  Decision Tree and Naive Bayes. A mismatch in it weighs 5.0 in the KNN distance,
  and other category mismatches weigh 1.0.
- `Loan_Status = Y` is the positive class for precision, recall and F1.
- The benchmark reports the mean of the metrics and the median of the times over
  5 seeded trials. Prediction time covers the whole test set.
- The app's Prediction page trains on all rows. The benchmark trains on 80% only.

## Limitations
- Missing values are filled using the whole dataset before splitting (as the project
  specification describes), so the test rows slightly influence the fill values.
- `TotalIncome` is derived from two other income columns, so Naive Bayes's assumption
  that features are independent does not fully hold.
- `Gender` and `Married` are used because they are in the dataset. A real lending
  system should not decide on such attributes without a fairness and legal review.

## Group members
[Member 1], [Member 2], [Member 3], [Member 4], [Member 5]
