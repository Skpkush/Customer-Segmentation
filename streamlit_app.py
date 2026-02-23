"""
streamlit_app.py
-----------------
Interactive Customer Segmentation Dashboard
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44BBA4', '#E94F37', '#393E41']


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Transaction CSV",
    type=['csv'],
    help="Required columns: CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice"
)

n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4)
avg_margin = st.sidebar.slider("Profit Margin (%)", min_value=5, max_value=40, value=20) / 100

st.sidebar.markdown("---")
st.sidebar.info("""
**Required Columns:**
- CustomerID
- InvoiceNo
- InvoiceDate
- Quantity
- UnitPrice
""")


# ── Main Header ────────────────────────────────────────────────────────────────
st.title("🛒 Advanced Customer Segmentation Dashboard")
st.markdown("> RFM Analysis + Machine Learning Clustering | Portfolio Project")
st.markdown("---")


# ── Load or Demo Data ──────────────────────────────────────────────────────────
@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n = 30000
    customer_ids = np.random.randint(12000, 16000, 3000)
    records = []
    for _ in range(n):
        cid = np.random.choice(customer_ids)
        date = pd.Timestamp('2010-12-01') + pd.Timedelta(days=np.random.randint(0, 365))
        records.append({
            'InvoiceNo': f'5{np.random.randint(10000, 99999)}',
            'Quantity': np.random.choice([1, 2, 3, 6], p=[0.5, 0.25, 0.15, 0.1]),
            'InvoiceDate': date,
            'UnitPrice': round(np.random.lognormal(1.5, 0.8), 2),
            'CustomerID': float(cid)
        })
    return pd.DataFrame(records)


@st.cache_data
def process_data(df_raw, n_clusters, avg_margin):
    df = df_raw.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

    snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalAmount', 'sum')
    ).reset_index()

    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    def segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4:    return 'Champions'
        elif r >= 3 and f >= 3:             return 'Loyal Customers'
        elif r >= 4 and f <= 2:             return 'Potential Loyalists'
        elif r >= 3 and m >= 4:             return 'Big Spenders'
        elif r <= 2 and f >= 3 and m >= 3:  return 'At Risk'
        elif r <= 2 and f >= 4:             return 'Cannot Lose Them'
        else:                               return 'Lost Customers'

    rfm['RFM_Segment'] = rfm.apply(segment, axis=1)

    rfm['CLV'] = (rfm['Monetary'] / rfm['Frequency'].clip(1) *
                  rfm['Frequency'] / (rfm['Recency'].clip(1) / 30) * avg_margin * 12).round(2)

    # ML Clustering
    X = StandardScaler().fit_transform(np.log1p(rfm[['Recency', 'Frequency', 'Monetary']]))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['ML_Cluster'] = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    rfm['PC1'] = X_pca[:, 0]
    rfm['PC2'] = X_pca[:, 1]

    sil = silhouette_score(X, rfm['ML_Cluster'])
    return rfm, sil, pca.explained_variance_ratio_


# Load data
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.success(f"✅ File uploaded: {len(df_raw):,} rows")
else:
    st.info("📊 Using demo data. Upload your CSV to use real data.")
    df_raw = generate_demo_data()

rfm, sil_score, pca_var = process_data(df_raw, n_clusters, avg_margin)


# ── KPI Cards ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Customers", f"{len(rfm):,}")
col2.metric("💷 Total Revenue", f"£{rfm['Monetary'].sum():,.0f}")
col3.metric("🤖 Silhouette Score", f"{sil_score:.3f}")
col4.metric("📊 Avg CLV", f"£{rfm['CLV'].mean():,.0f}")

st.markdown("---")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 RFM Segments", "🤖 ML Clusters", "📈 Business Insights", "📋 Data Export"])

with tab1:
    st.subheader("RFM Business Segments")
    col_a, col_b = st.columns(2)

    with col_a:
        seg_counts = rfm['RFM_Segment'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
               colors=PALETTE[:len(seg_counts)], startangle=140)
        ax.set_title('Customer Distribution by RFM Segment', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col_b:
        rev_seg = rfm.groupby('RFM_Segment')['Monetary'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(rev_seg.index, rev_seg.values, color=PALETTE[:len(rev_seg)], edgecolor='black')
        ax.set_title('Revenue by Segment (£)', fontweight='bold')
        ax.tick_params(axis='x', rotation=35)
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()

    seg_table = rfm.groupby('RFM_Segment').agg(
        Customers=('CustomerID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
        Avg_CLV=('CLV', 'mean')
    ).round(1).sort_values('Total_Revenue', ascending=False)
    st.dataframe(seg_table, use_container_width=True)

with tab2:
    st.subheader(f"K-Means Clustering (K={n_clusters})")
    st.metric("Silhouette Score", f"{sil_score:.4f}", help=">0.5 = good, >0.7 = strong")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_clusters):
        mask = rfm['ML_Cluster'] == i
        ax.scatter(rfm.loc[mask, 'PC1'], rfm.loc[mask, 'PC2'],
                   c=PALETTE[i % len(PALETTE)], alpha=0.6, s=25, label=f'Cluster {i}')
    ax.set_title(f'K-Means Clusters (PCA 2D | Variance: {sum(pca_var)*100:.1f}%)', fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    ml_table = rfm.groupby('ML_Cluster').agg(
        Customers=('CustomerID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Revenue=('Monetary', 'mean'),
        Avg_CLV=('CLV', 'mean')
    ).round(1)
    st.dataframe(ml_table, use_container_width=True)

with tab3:
    st.subheader("Business Insights")

    col_x, col_y = st.columns(2)
    with col_x:
        clv_seg = rfm.groupby('RFM_Segment')['CLV'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(clv_seg.index, clv_seg.values, color=PALETTE[:len(clv_seg)], edgecolor='black')
        ax.set_title('Average CLV by Segment', fontweight='bold')
        ax.tick_params(axis='x', rotation=35)
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()

    with col_y:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(rfm['Frequency'], rfm['Monetary'], c=rfm['ML_Cluster'].map(
            lambda x: PALETTE[x % len(PALETTE)]), alpha=0.5, s=20)
        ax.set_title('Frequency vs Monetary (by Cluster)', fontweight='bold')
        ax.set_xlabel('Frequency'); ax.set_ylabel('Monetary (£)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

with tab4:
    st.subheader("📥 Download Segmented Customer List")
    st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'CLV',
                        'R_Score', 'F_Score', 'M_Score', 'RFM_Segment', 'ML_Cluster']].head(100),
                 use_container_width=True)

    csv = rfm.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Full Segment CSV",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Customer Segmentation Dashboard | Built with Streamlit + Scikit-learn | Portfolio Project 2026")