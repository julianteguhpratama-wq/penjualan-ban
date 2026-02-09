import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Clustering Penjualan Ban",
    layout="wide"
)

# ================== LOAD DATA ==================
df = pd.read_csv('databansk.csv', sep=';')

# Simpan kolom Ban
ban_col = df[['Ban']]

# Data numerik untuk K-Means
X = df.drop(['Ban'], axis=1)

# ================== TAMPIL DATA ==================
st.header("Isi Dataset")
st.dataframe(df, use_container_width=True)

st.header("Data Setelah Menghapus Atribut Non-Numerik")
st.dataframe(X, use_container_width=True)

# ================== ELBOW METHOD ==================
st.header("Cluster Optimal Berdasarkan Metode Elbow")

clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X)
    clusters.append(km.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(12, 8))
sns.lineplot(
    x=list(range(1, 11)),
    y=clusters,
    marker='o',
    ax=ax_elbow
)

ax_elbow.set_title('Metode Elbow')
ax_elbow.set_xlabel('Jumlah Cluster (k)')
ax_elbow.set_ylabel('Inertia')

# Panah elbow
ax_elbow.annotate(
    'Possible Elbow Point',
    xy=(2, 700000),
    xytext=(2, 2000),
    xycoords='data',
    arrowprops=dict(arrowstyle='->', color='blue', lw=2)
)

ax_elbow.annotate(
    'Possible Elbow Point',
    xy=(4, 190000),
    xytext=(4, 600000),
    xycoords='data',
    arrowprops=dict(arrowstyle='->', color='blue', lw=2)
)

st.pyplot(fig_elbow)

# ================== SLIDER JUMLAH CLUSTER ==================
st.sidebar.header("Pengaturan Cluster")
clust = st.sidebar.slider(
    "Pilih Jumlah Cluster (K)",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)

# ================== FUNGSI K-MEANS + LABEL STABIL ==================
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

    X_plot['Label'] = [label_mapping[l] for l in labels_raw]

    # ================== VISUALISASI CLUSTER ==================
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6))

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

    ax_cluster.set_title('Posisi Data Berdasarkan Cluster')
    ax_cluster.set_xlabel('Harga')
    ax_cluster.set_ylabel('Terjual')

    st.header("Visualisasi Hasil Clustering")
    st.pyplot(fig_cluster, use_container_width=True)

    # ================== TABEL AKHIR ==================
    hasil_akhir = pd.concat(
        [
            ban_col.reset_index(drop=True),
            X_plot.reset_index(drop=True)
        ],
        axis=1
    )

    st.subheader("Hasil Akhir K-Means Clustering")
    st.dataframe(hasil_akhir, use_container_width=True)

# ================== EKSEKUSI ==================
k_means_label_fix(clust)

