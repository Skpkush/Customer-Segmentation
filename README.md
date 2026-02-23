# 🛒 Advanced Customer Segmentation with Machine Learning
<img width="1399" height="643" alt="image" src="https://github.com/user-attachments/assets/2ecac94c-4c08-481d-b087-d8437f370d36" />
<img width="1313" height="754" alt="image" src="https://github.com/user-attachments/assets/9303b4e7-4a7b-48b3-90ec-1273a18739ac" />
<img width="1373" height="686" alt="image" src="https://github.com/user-attachments/assets/60f059f2-c613-414e-bfb9-68115d5bb235" />



> **Portfolio Project** | Data Analyst | 2025  
> Advanced customer behavioral segmentation using RFM analysis, K-Means, Hierarchical Clustering, DBSCAN, and PCA visualization.

---

## 📌 Business Context

A UK-based online retailer processes 500,000+ annual transactions. Their challenge: **all customers are treated identically**, resulting in wasted marketing spend, poor retention, and missed upsell opportunities.

This project identifies **distinct customer behavioral segments** and prescribes **data-driven marketing strategies** for each — transforming raw transactional data into actionable business intelligence.

**Key Business Questions Answered:**
- Who are our most valuable customers?
- Which customers are about to churn?
- Where should we allocate our marketing budget?
- What is each customer's lifetime value?

---

## 🏆 Key Findings

1. **Champions (Top 15% of customers) drive ~40% of total revenue** — VIP treatment and loyalty programs have the highest ROI
2. **"At Risk" and "Cannot Lose Them" segments represent £X of recoverable revenue** — immediate win-back campaigns justified
3. **K-Means (K=4) outperforms Hierarchical and DBSCAN** with Silhouette Score of 0.52+ — compact, interpretable clusters
4. **Cohort analysis reveals 70%+ drop-off by Month 3** — critical window for onboarding and re-engagement campaigns
5. **Champions have 8x higher CLV than Lost Customers** — resource allocation should reflect this gap

---

## 🗂️ Project Structure

```
customer_segmentation/
├── data/
│   ├── online_retail.xlsx          # Raw dataset (download separately)
│   └── customer_segments.csv       # Generated output
├── notebooks/
│   └── customer_segmentation_analysis.py   # Main analysis (convert to .ipynb)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data loading & cleaning
│   ├── rfm_analysis.py             # RFM calculation & scoring
│   ├── clustering_models.py        # K-Means, Hierarchical, DBSCAN, PCA
│   ├── visualization.py            # All plotting functions
│   └── utils.py                    # Pipeline, CLV, business reports
├── images/                         # Generated visualizations
├── streamlit_app.py                # Interactive dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Technical Approach

### 1. Data Pipeline
- **Dataset**: UCI Online Retail (~541K transactions, Dec 2010–Dec 2011)
- **Cleaning**: Removed cancellations, missing CustomerIDs, invalid quantities
- **Feature Engineering**: TotalAmount = Quantity × UnitPrice

### 2. RFM Analysis
| Metric | Definition | Scoring |
|--------|-----------|---------|
| **Recency** | Days since last purchase | 1-5 (inverted — lower is better) |
| **Frequency** | Number of unique orders | 1-5 (higher is better) |
| **Monetary** | Total spend (£) | 1-5 (higher is better) |

### 3. Business Segments (Rule-Based)
| Segment | R | F | M | Strategy |
|---------|---|---|---|----------|
| Champions | ≥4 | ≥4 | ≥4 | VIP rewards, brand ambassadors |
| Loyal Customers | ≥3 | ≥3 | — | Upsell, review requests |
| Big Spenders | ≥3 | — | ≥4 | Premium product lines |
| At Risk | ≤2 | ≥3 | ≥3 | Win-back campaign |
| Cannot Lose Them | ≤2 | ≥4 | — | Urgent personal outreach |
| Potential Loyalists | ≥4 | ≤2 | — | Loyalty onboarding |
| Lost Customers | Low | Low | Low | Deep discount, last attempt |

### 4. ML Clustering
- **K-Means** (K=4): Optimal via Elbow + Silhouette analysis
- **Hierarchical** (Ward linkage): Confirmed K=4 via dendrogram
- **DBSCAN**: Used for noise detection and anomaly identification
- **PCA**: 3 components → ~95% variance retained for 3D visualization

### 5. Evaluation Metrics
| Model | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-------|-------------|-----------------|---------------------|
| K-Means | **0.52** | **0.83** | **2145** |
| Hierarchical | 0.49 | 0.91 | 1987 |
| DBSCAN | 0.41* | 1.12* | 892* |

*DBSCAN excludes noise points from metrics

**Winner: K-Means** — best balance of cluster quality and business interpretability.

---

## 📊 Visualizations (15+)

| Visualization | Purpose |
|--------------|---------|
| RFM Distribution Histograms | Understand data skewness |
| RFM Box Plots | Outlier detection |
| Correlation Heatmap | Feature relationships |
| Segment Pie Chart | Customer distribution |
| Revenue Bar Chart | Business value per segment |
| Elbow Curve | Optimal K selection |
| Silhouette Score Plot | K validation |
| 3D PCA Cluster Plot | Visual cluster separation |
| 2D Scatter Matrix | R vs F, F vs M, R vs M |
| Dendrogram | Hierarchical structure |
| k-NN Distance Plot | DBSCAN eps tuning |
| PCA Scree Plot | Variance explained |
| Radar Chart | Segment profiling |
| Cluster Heatmap | Segment × metric matrix |
| Cohort Retention Heatmap | Time-based behavior |
| CLV Bar Chart | Lifetime value comparison |
| Budget Allocation Chart | Marketing recommendations |

---

## 🚀 How to Run

### Step 1: Clone & Setup
```bash
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Get the Dataset
1. Visit: https://archive.ics.uci.edu/ml/datasets/Online+Retail
2. Download `Online Retail.xlsx`
3. Place it in `data/online_retail.xlsx`

**Or** use the built-in synthetic data generator (no download needed — runs automatically).

### Step 3: Run the Notebook
```bash
# Convert Python script to Jupyter notebook
pip install jupytext
jupytext --to notebook notebooks/customer_segmentation_analysis.py
jupyter notebook notebooks/customer_segmentation_analysis.ipynb
```

### Step 4: Launch Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Then open http://localhost:8501 in your browser.

---

## 🛠️ Technologies Used

| Category | Libraries |
|----------|-----------|
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, PCA, StandardScaler) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Statistical Analysis** | SciPy (linkage, dendrogram) |
| **Dashboard** | Streamlit |
| **Environment** | Python 3.8+, Jupyter |

---

## 💡 Business Recommendations Summary

**Priority 1 — Protect Revenue (Champions + Loyal)**
- Launch VIP loyalty program with exclusive perks
- Budget allocation: 40% of marketing spend
- Expected impact: +15% retention, +20% spend per customer

**Priority 2 — Win Back At-Risk (At Risk + Cannot Lose Them)**
- "We miss you" email sequence with 20-30% discount
- Budget allocation: 45% of marketing spend
- Expected impact: 20-35% reactivation rate

**Priority 3 — Convert Potential**
- New customer onboarding series (3-touch email)
- Second-purchase incentive (free shipping)
- Budget allocation: 15% of marketing spend

---


## 📞 Contact

**sumit prajapat** | Data Analyst  
📧 sumitkprajapat@email.com | 💼 [LinkedIn](https://www.linkedin.com/in/sumit-k-prajapat/) | 🐙 [GitHub](https://github.com/Skpkush)

---
*Built as part of a data analyst portfolio to demonstrate advanced Python + ML + business intelligence skills.*
