import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Clustering Penjualan Ban", layout="wide")

# ================== LOAD DATA ==================
df = pd.read_csv('databansk.csv', sep=';')

# Simpan kolom Ban
ban_col = df[['Ban']]

# Data numerik untuk K-Means
X = df.drop(['Ban'], axis=1)

# ================== TAMPIL DATA ==================
st.header("Isi Dataset Awal")
st.dataframe(df)

st.header("Data Setelah Menghapus Atribut Non-Numerik")
st.dataframe(X)

# ================== ELBOW METHOD ==================
# ================== ELBOW METHOD ==================
st.header("Penentuan Jumlah Cluster Optimal (Metode Elbow)")

clusters = []
K = range(1, 11)

for i in K:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X)
    clusters.append(km.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(12, 8))

sns.lineplot(
    x=K,
    y=clusters,
    marker='o',
    ax=ax_elbow
)

ax_elbow.set_title('Grafik Metode Elbow', fontsize=14)
ax_elbow.set_xlabel('Jumlah Cluster (k)')
ax_elbow.set_ylabel('Inertia')

# ===== PENANDA TITIK ELBOW (CONTOH k = 3) =====
elbow_k = 3
ax_elbow.scatter(
    elbow_k,
    clusters[elbow_k - 1],
    color='red',
    s=150,
    zorder=5
)

ax_elbow.annotate(
    f'Elbow Point (k = {elbow_k})',
    xy=(elbow_k, clusters[elbow_k - 1]),
    xytext=(elbow_k + 1, clusters[elbow_k - 1] * 1.1),
    arrowprops=dict(arrowstyle='->', color='red', lw=2),
    fontsize=12
)

st.pyplot(fig_elbow)

# ================== SLIDER JUMLAH CLUSTER ==================
st.sidebar.header("Pengaturan Clustering")
clust = st.sidebar.slider(
    "Pilih Jumlah Cluster (k)",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)

# ================== FUNGSI K-MEANS + FIX LABEL ==================
def k_means_label_fix(n_clust):
    X_plot = X.copy()

    kmeans = KMeans(n_clusters=n_clust, random_state=42)
    kmeans.fit(X_plot)

    labels_raw = kmeans.labels_

    # Hitung rata-rata Terjual tiap cluster
    cluster_mean = (
        X_plot.assign(Labels=labels_raw)
        .groupby('Labels')['Terjual']
        .mean()
        .sort_values(ascending=False)
    )

    # Mapping ulang label (1 = paling laris)
    label_mapping = {
        old_label: new_label + 1
        for new_label, old_label in enumerate(cluster_mean.index)
    }

    # Simpan label final yang stabil
    X_plot['Label'] = [label_mapping[l] for l in labels_raw]

    # ================== VISUALISASI CLUSTER ==================
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 8))

    sns.scatterplot(
        data=X_plot,
        x='Harga',
        y='Terjual',
        hue='Label',
        size='Label',
        palette=sns.color_palette('hls', n_clust),
        ax=ax_cluster
    )

    for label in sorted(X_plot['Label'].unique()):
        ax_cluster.annotate(
            f'Cluster {label}',
            (
                X_plot[X_plot['Label'] == label]['Harga'].mean(),
                X_plot[X_plot['Label'] == label]['Terjual'].mean()
            ),
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold',
            color='black'
        )

    ax_cluster.set_title('Visualisasi Hasil K-Means Clustering')
    ax_cluster.set_xlabel('Harga')
    ax_cluster.set_ylabel('Terjual')

    st.header("Visualisasi Hasil Clustering")
    st.pyplot(fig_cluster)

    # ================== TABEL AKHIR (DENGAN KOLOM BAN) ==================
    hasil_akhir = pd.concat(
        [
            ban_col.reset_index(drop=True),
            X_plot.reset_index(drop=True)
        ],
        axis=1
    )

    st.subheader("Tabel Hasil Akhir K-Means Clustering")
    st.dataframe(hasil_akhir)

# ================== EKSEKUSI ==================
k_means_label_fix(clust)
