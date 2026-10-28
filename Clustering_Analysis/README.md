# Part 2: Customer Segmentation (K-Means Clustering)

**Team 8 | Customer Churn Analysis for a Telecommunications Company**
**Owner:** Jyoti Saini, Data Analyst (Clustering)
**Jira:** epic T8-27 Stage 2 Clustering results, stories T8-46 to T8-52 (C1 to C7)

## Purpose

Segment the client's 7,043 customers into meaningful groups with K-Means, profile each group, and compare churn across the groups, so that the ANN model (Part 3) and the retention strategies (Part 4) can target the customers most likely to leave.

## Folder structure

```
02_Clustering_Jyoti_Saini/
├── README.md
├── requirements.txt
├── data/
│   ├── Dataset_ATS_v2.csv              original client file (used only to describe segments in real units)
│   └── Updated_Dataset.csv             processed dataset from Part 1 (Data Engineer)
├── notebooks/
│   └── 02_clustering_kmeans.ipynb      full clustering workflow with outputs
├── src/
│   └── 02_clustering_kmeans.py         same workflow as a plain Python script
├── outputs/
│   ├── data/
│   │   ├── Updated_Dataset_with_Clusters.csv   Part 1 dataset + Cluster column (for the ANN)
│   │   ├── Clustered_Customers.csv             every customer with Cluster and Segment name
│   │   └── Cluster_Profiles.csv                one row per segment: size, tenure, charges, churn, mix
│   ├── models/
│   │   └── kmeans_k4_model.joblib             trained K-Means model + feature list + segment names
│   └── figures/
│       ├── fig1_elbow_silhouette.png
│       ├── fig2_pca_clusters.png
│       ├── fig3_tenure_charges_clusters.png
│       ├── fig4_churn_by_cluster.png
│       └── fig5_segment_profiles.png
└── docs/
    └── Jyoti_Saini_Stage2_Clustering_Results_Report.docx
```

## How to run

```
pip install -r requirements.txt
cd notebooks
jupyter notebook 02_clustering_kmeans.ipynb      # or: cd src && python 02_clustering_kmeans.py
```

Inputs are read from `data/` and all results are written to `outputs/`. The model uses a fixed random state, so every run gives the same clusters.

## Method

1. **Readiness check (C1):** 7,043 rows, no missing values, consistent labels. 302 duplicate rows kept (no customer ID).
2. **Feature selection (C2):** 9 features: tenure, MonthlyCharges, SeniorCitizen, Dependents, PhoneService, MultipleLines, Fibre internet, One-year and Two-year contract. Churn held back; gender excluded (churn 26.9% vs 26.2%).
3. **Encoding and scaling (C3):** text and boolean columns converted to 0/1; tenure and MonthlyCharges standardised (mean 0, SD 1) in Part 1.
4. **Number of clusters (C4):** elbow method and silhouette score for k = 1 to 10; k = 4 chosen.
5. **Final model (C5):** K-Means, k = 4, k-means++, n_init = 10, random_state = 42. Silhouette 0.225; stability check ARI 0.999 to 1.0.
6. **Profiling (C6):** PCA plot (62% of variance in two components) and segment profiles.
7. **Model file:** saved to `outputs/models/kmeans_k4_model.joblib` (reload check passed).
8. **Handover (C7):** outputs below delivered on 1 November 2026.

## Using the saved model

```python
import joblib, pandas as pd
pkg = joblib.load("outputs/models/kmeans_k4_model.joblib")
X = df[pkg["features"]]                      # same 9 prepared features, same scaling
clusters = pd.Series(pkg["model"].predict(X)).map(pkg["relabel"])
segments = clusters.map(pkg["segment_names"])
```

## Results

| Cluster | Segment | Customers | Avg tenure | Avg monthly charge | Churn rate |
|---|---|---|---|---|---|
| C0 | New Budget Customers | 1,721 (24.4%) | 10 months | $32 | 24.6% |
| C1 | At-Risk New Premium Customers | 2,223 (31.6%) | 15 months | $81 | 49.0% |
| C2 | Loyal Budget Customers | 1,176 (16.7%) | 54 months | $35 | 4.8% |
| C3 | Loyal Premium Customers | 1,923 (27.3%) | 59 months | $93 | 15.5% |

Overall churn is 26.5%. C1 holds 58.3% of all churners and about two thirds of the monthly revenue lost to churn, so it is the priority group for retention. Contract type and internet service are spread evenly across all four segments.

## Handover

| For | File |
|---|---|
| Part 3, ANN (Bao) | `outputs/data/Updated_Dataset_with_Clusters.csv` (note: gender and Dependents are text and need converting to 0/1) |
| Part 4, Business analysis (Fahad) | `outputs/data/Clustered_Customers.csv`, `outputs/data/Cluster_Profiles.csv` |
| Stage 2 deliverables (Raman) | `docs/Jyoti_Saini_Stage2_Clustering_Results_Report.docx`, `outputs/figures/` |

## Limitations

Single snapshot of ten attributes; no usage, billing history or payment method data. K-Means assumes roughly round, similar-sized clusters, so segments are broad customer groups rather than exact boundaries.
