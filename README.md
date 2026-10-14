# Customer Churn Prediction and RFM Segmentation

This project predicts which active customers of an online wholesaler will stop buying in the next 90 days, groups them into RFM segments, and produces a ranked call list for the retention team.

> This rebuilds, on public data, the churn and segmentation work I did as a Data Analyst at Think and Wash. The original work used internal company data that I can't share, so the metrics here come from the public data and differ from the original project's.

![Results](results/churn.png)

## Problem

A retention team can only contact a fraction of its customers, so it needs answers to three questions:
- Who is likely to stop ordering?
- Which of them are worth the effort?
- What kind of offer fits each group?

The current rule of thumb is "no order in 90 days = churned". It catches customers only after they're already gone.

## Data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (CC BY 4.0) has 1.07M invoice lines from a UK online gift wholesaler, Dec 2009 – Dec 2011, across 5,875 identified customers in 41 countries.

**Cleaning:**
- **Duplicates:** 34K exact duplicate lines removed. 1–9 Dec 2010 appears in both Excel sheets, and some lines were scanned twice.
- **No customer ID:** 235K lines (23%) are guest checkouts. They can't be tracked over time, so they're dropped.
- **Non-product lines:** postage, fees, manual adjustments, bad-debt write-offs and test items, which aren't purchases.
- **Net spend:** spend is measured after returns. One customer ordered £168K and cancelled the whole order the same day. Counted gross, that order would have topped the call list.

## Approach

**1. A leakage-safe churn label.** Each snapshot takes a cutoff date and the customers who bought in the 180 days before it.
- **Features** use only those 180 days.
- **Label:** `churned = 1` if the customer places no order in the next 90 days.
- [test_churn.py](test_churn.py) checks the rule: transactions after the cutoff never change a feature.

**2. An out-of-time test.** Snapshots are taken every 90 days.
- **Test:** the last one with a complete label window (cutoff 10 Sep 2011, 2,783 customers).
- **Train:** the five quarters before it (14,869 customer-snapshots).
- **Cross-validation:** folds are grouped by customer, because the same customer appears in several snapshots.

**3. Seasonality.** Churn depends on the season. Customers stock up before Christmas, so churn in the Sep–Dec window is 38–39%, against 50–65% in the other quarters. The training data includes the same quarter a year earlier, plus a `peak_season` flag, so the model doesn't over-flag customers in peak season. My first version lacked this, and its accuracy was below the simple rule.

**4. Models.** Logistic regression, Random Forest and XGBoost run in sklearn pipelines (imputation, plus log-scaling for LogReg) on 11 features: recency, tenure, order counts, net spend, average order value, product range, gap between orders, return rate, UK flag and season. Model selection uses cross-validated ROC-AUC. The decision threshold is the cross-validated F1 optimum, fixed before the test.

**5. Segmentation.** K-means runs on log-scaled recency, frequency and monetary value. I used k = 4: silhouette is highest at k = 2, but that only splits active from lapsed customers. Among k ≥ 3, k = 4 scores best, and each segment maps to a different action.

## Results

Out-of-time test, Sep–Dec 2011 (38.2% churned):

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy | Precision, top 20% |
|---|---|---|---|---|---|---|
| Always predict churn | 0.50 | 0.55 | 0.38 | 1.00 | 0.38 | 0.37 |
| Rule: no order in 90 days | 0.66 | 0.49 | 0.54 | 0.45 | 0.64 | 0.59 |
| Logistic regression | 0.73 | 0.62 | 0.53 | 0.76 | 0.65 | 0.60 |
| Random Forest | 0.74 | 0.62 | 0.54 | 0.73 | 0.66 | 0.64 |
| **XGBoost (chosen by CV)** | **0.74** | **0.61** | **0.56** | **0.65** | **0.67** | **0.65** |

In cross-validation on the training quarters, XGBoost scored ROC-AUC 0.76, F1 0.75 and accuracy 0.69. Accuracy isn't the headline metric here: predicting "stays" for everyone would score 62% on the test quarter.

**Findings:**
- **Most churners still look active.** 590 of the 1,064 test churners (55%) had ordered in the last 90 days, so the rule can never flag them. The model catches 307 of them (52%), and 51% of the customers it flags in that group really do churn.
- **Order rhythm and tenure matter most.** By permutation importance, the strongest signals are the usual gap between orders and how long the customer has been buying, ahead of recency. Customers first seen less than 6 months ago churn at about 63%, against 42% for established ones.
- **The models are close.** The three models are within 0.01 AUC of each other. Logistic regression is a reasonable choice if explainability matters more than the last point of AUC.

**Segments (test quarter):**

| Segment | Customers | Median recency | Median orders (6 mo) | Share of net spend | Actual churn | Action |
|---|---|---|---|---|---|---|
| Champions | 301 | 8 days | 8 | 53% | 5% | Protect: account manager, early access |
| Regulars | 887 | 44 days | 3 | 32% | 26% | **Call list:** largest spend at risk |
| Recent, low spend | 337 | 7 days | 2 | 6% | 39% | Second-order offer. A recency rule calls them safe. |
| Lapsing | 1,258 | 103 days | 1 | 10% | 55% | Cheap automated win-back email |

**Scoring.** [score.py](score.py) scores the 3,471 customers active at the end of the data. It ranks the at-risk customers by expected spend at risk (churn probability × 6-month net spend) and writes `results/scores.csv`. Regulars hold the largest expected spend at risk, about £517K.

## How to run

Tested with Python 3.12. Download `online_retail_II.xlsx` from the UCI link above into `data/`.

```bash
pip install -r requirements.txt
python test_churn.py   # leakage check, prints "ok"
python train.py        # ~15 s after the first run, which spends ~45 s converting the xlsx to CSV
python score.py        # today's call list -> results/scores.csv
```

[notebooks/eda.ipynb](notebooks/eda.ipynb) walks through cleaning, the seasonality problem, churn drivers, where the model beats the rule, and the segments.

## Files

| File | What it does |
|---|---|
| [train.py](train.py) | Loads and cleans the data, builds snapshots, cross-validates 3 models against baselines, runs the out-of-time test and segmentation, and saves `models/churn.joblib` and `results/` |
| [score.py](score.py) | Scores current customers and writes the ranked call list |
| [test_churn.py](test_churn.py) | Checks the no-leakage rule for features and labels |
| [notebooks/eda.ipynb](notebooks/eda.ipynb) | Analysis and reasoning behind each choice |

## What I'd do next

- **Measure uplift, not just risk.** Run an A/B test on the call list. A customer at high risk isn't necessarily one a call can save.
- **Recalibrate every quarter.** Churn levels shift with the season. Retrain each quarter and track predicted vs actual churn by segment.
- **Add product-level signals.** Category mix and price sensitivity could help with wholesale buyers.
