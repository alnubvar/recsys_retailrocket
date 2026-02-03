# Recommendation System for RetailRocket (Applied ML)

> **Applied recommender systems case study**
> Building and evaluating candidate-generation and ranking pipelines
> on real-world e-commerce interaction data.

Portfolio-ready applied machine learning project that builds and evaluates a real-world e-commerce recommender system on the RetailRocket dataset. The focus is practical decision-making, model comparison, and business interpretation rather than purely academic benchmarks.

---

## Overview

Goal: recommend Top-N products for each user with the highest probability of purchase.

Constraints:
- extremely sparse implicit feedback
- no user profiles
- cold start for new users and items

---

## Approaches Implemented

1. Random baseline
2. Popularity baseline
3. Item-based Collaborative Filtering (co-occurrence)
4. Matrix Factorization (ALS for implicit feedback)
5. Feature-based Ranking (Learning to Rank)

---

## Evaluation

Metrics:
- `precision@k`
- `recall@k`

Test split is based on actual user purchases.

---

## Key Results

Observations:
- Popularity baseline is surprisingly strong and stable on sparse e-commerce data.
- Item-based CF and ALS do not outperform popularity due to limited user histories and low purchase frequency.
- Feature-based ranking delivers the best quality.

Sample metrics:

| k | precision@k | recall@k |
|---|-------------|----------|
| 5 | ~0.26 | ~0.76 |
| 10 | ~0.17 | ~0.92 |
| 20 | ~0.10 | ~0.96 |

---

## Feature Importance (Ranking Model)

RandomForest feature importance:
- `item_pop` ~66%
- `user_activity` ~16%
- `ui_interactions` ~9%
- `ui_max_weight` ~9%

Interpretation: the model relies primarily on global demand, then refines recommendations using user behavior and user-item interaction history.

---

## Suggested Production Architecture

1. Candidate generation
   - popularity-based recommendations
   - optional item-based CF for diversity
2. Ranking
   - feature-based model (RandomForest or Gradient Boosting)
3. Cold start
   - new users: popularity
   - new items: controlled exposure and feedback collection

This pipeline scales well, is robust to sparsity, and aligns with production best practices.

---

## Project Structure

```
RECSYS_RETAILROCKET/
├── data/
│ ├── raw/                # RetailRocket raw data
│ └── processed/          # train / val / test
├── notebooks/
│ ├── 00_dataset_overview.ipynb
│ ├── 01_eda_events.ipynb
│ ├── 02_data_preparation.ipynb
│ ├── 03_baseline_popularity.ipynb
│ ├── 04_item_based_cf.ipynb
│ ├── 05_matrix_factorization_als.ipynb
│ ├── 06_model_comparison.ipynb
│ ├── 07_ranking_with_features.ipynb
│ └── 08_business_insights.ipynb
├── src/
│ └── data/
│ └── preprocessing.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Limitations and Next Steps

- add temporal and session features
- evaluate stronger rankers (LightGBM, XGBoost)
- explore neural candidate generation
- add online metrics and A/B testing plan

---

## Key Takeaway

In highly sparse e-commerce settings,
strong baselines and feature-based ranking
often outperform complex collaborative filtering models.
Practical recommender systems should prioritize robustness,
scalability, and business impact over algorithmic complexity.

---

## Summary

This project delivers a full applied ML pipeline for a real e-commerce recommender system:
- data exploration and constraint analysis
- baseline and collaborative filtering comparison
- feature-based ranking model
- business-focused interpretation and production recommendations

Version: v1.0, portfolio-ready.
