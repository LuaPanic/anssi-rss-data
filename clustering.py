"""
comparatif clustering — profil de dangerosité des CVE ANSSI
features : cvss, log(epss), epss_percentile, nb_documents, sévérité
sortie : output/clustering.png
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    MiniBatchKMeans, AffinityPropagation, MeanShift, estimate_bandwidth,
    SpectralClustering, AgglomerativeClustering,
    DBSCAN, HDBSCAN, OPTICS, Birch,
)
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

warnings.filterwarnings("ignore")

print("chargement des données...")
df = pd.read_csv("output/cve_enriched.csv")
df["cvss"]           = pd.to_numeric(df["cvss"], errors="coerce")
df["epss"]           = pd.to_numeric(df["epss"], errors="coerce")
df["epss_percentile"]= pd.to_numeric(df["epss_percentile"], errors="coerce")
df["nb_documents"]   = pd.to_numeric(df["nb_documents"], errors="coerce")

sev_map = {"Faible": 1, "Moyenne": 2, "Élevée": 3, "Critique": 4}
df["sev_num"] = df["severite"].map(sev_map)

# features retenues : cvss, log-epss, percentile epss, nb_documents, sévérité
df["log_epss"] = np.log1p(df["epss"])  # log(1+x) pour gérer le skew extrême

features = ["cvss", "log_epss", "epss_percentile", "nb_documents", "sev_num"]
df_clean = df[features].dropna()
df_clean = df_clean.sample(n=min(2000, len(df_clean)), random_state=42)

X = df_clean.values
X_scaled = StandardScaler().fit_transform(X)

# réduction 2D pour la visualisation (PCA puis on affiche les 2 premiers axes)
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_
print(f"PCA : PC1={var[0]:.1%}  PC2={var[1]:.1%}  total={sum(var):.1%} de variance expliquée")

# -----------------------------------------------------------------------
# algorithmes
# -----------------------------------------------------------------------
algos = [
    ("MiniBatch\nKMeans",         MiniBatchKMeans(n_clusters=4, random_state=42, n_init="auto")),
    ("Affinity\nPropagation",     AffinityPropagation(damping=0.85, random_state=42, max_iter=400)),
    ("MeanShift",                 MeanShift(bandwidth=estimate_bandwidth(X_scaled, quantile=0.3, n_samples=500), n_jobs=-1)),
    ("Spectral\nClustering",      SpectralClustering(n_clusters=4, random_state=42, n_jobs=-1)),
    ("Ward",                      AgglomerativeClustering(n_clusters=4, linkage="ward")),
    ("Agglomerative\nClustering", AgglomerativeClustering(n_clusters=4, linkage="average")),
    ("DBSCAN",                    DBSCAN(eps=0.5, min_samples=15)),
    ("HDBSCAN",                   HDBSCAN(min_cluster_size=30)),
    ("OPTICS",                    OPTICS(min_samples=20, xi=0.05, n_jobs=-1)),
    ("BIRCH",                     Birch(n_clusters=4)),
    ("Gaussian\nMixture",         GaussianMixture(n_components=4, random_state=42)),
]

CMAP = plt.cm.tab10

# -----------------------------------------------------------------------
# figure
# -----------------------------------------------------------------------
n_cols = 6
n_rows = 2
fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.0, n_rows * 4.0), facecolor="#f5f5f5")
axes = [axes_grid[r][c] for r in range(n_rows) for c in range(n_cols)][:len(algos)]
# masquer la dernière cellule vide si impair
for ax in axes_grid.flat:
    ax.set_visible(False)
for ax in axes:
    ax.set_visible(True)

fig.suptitle(
    f"comparatif clustering — profil de dangerosité CVE ANSSI (n={len(X_2d)}, "
    f"PCA 2D : {sum(var):.0%} variance)\n"
    f"features : cvss · log(epss) · epss_percentile · nb_documents · sévérité",
    fontsize=12, fontweight="bold", y=1.02
)

print(f"lancement de {len(algos)} algorithmes...")
for ax, (name, algo) in zip(axes, tqdm(algos, desc="clustering", unit="algo")):
    t0 = time.time()
    try:
        if isinstance(algo, GaussianMixture):
            algo.fit(X_scaled)
            labels = algo.predict(X_scaled)
        else:
            labels = algo.fit_predict(X_scaled)
    except Exception as e:
        labels = np.zeros(len(X_2d), dtype=int)
    elapsed = time.time() - t0

    n_clusters = len(set(labels) - {-1})
    n_noise    = int((labels == -1).sum())

    colors = ["#111111" if l == -1 else CMAP(l % 10) for l in labels]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=5, alpha=0.5, linewidths=0)

    ax.set_title(name, fontsize=8.5, fontweight="bold", pad=5)
    ax.set_xlabel(f"PC1 ({var[0]:.0%})", fontsize=7)
    if ax is axes[0]:
        ax.set_ylabel(f"PC2 ({var[1]:.0%})", fontsize=7)
    else:
        ax.set_ylabel("")

    info = f"{elapsed:.2f}s · {n_clusters} clusters"
    if n_noise:
        info += f" · {n_noise} bruit"
    ax.text(0.97, 0.03, info, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color="#444444")

    ax.set_facecolor("#fafafa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6)

plt.tight_layout()
out = "output/clustering.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"sauvegardé : {out}")
