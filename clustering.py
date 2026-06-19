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
from sklearn.metrics import silhouette_score, davies_bouldin_score
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



hdb = HDBSCAN(min_cluster_size=30)
labels_hdb = hdb.fit_predict(X_scaled)
n_clusters_hdb = len(set(labels_hdb) - {-1})
n_noise_hdb    = int((labels_hdb == -1).sum())

# métriques (hors bruit)
mask_valid = labels_hdb != -1
sil = silhouette_score(X_scaled[mask_valid], labels_hdb[mask_valid]) if mask_valid.sum() > 1 else float("nan")
dbi = davies_bouldin_score(X_scaled[mask_valid], labels_hdb[mask_valid]) if mask_valid.sum() > 1 else float("nan")
print(f"  clusters : {n_clusters_hdb}  |  bruit : {n_noise_hdb}  |  silhouette : {sil:.3f}  |  Davies-Bouldin : {dbi:.3f}")

# profil moyen par cluster
df_hdb = df_clean.copy()
df_hdb["cluster"] = labels_hdb
df_hdb_clean = df_hdb[df_hdb["cluster"] != -1]

profils = df_hdb_clean.groupby("cluster")[["cvss", "log_epss", "epss_percentile", "nb_documents", "sev_num"]].mean()
print("\nprofil moyen par cluster :")
print(profils.round(3).to_string())

# -----------------------------------------------------------------------
# figure analyse HDBSCAN
# -----------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), facecolor="#f5f5f5")
fig2.suptitle(
    f"analyse HDBSCAN — CVE ANSSI  |  {n_clusters_hdb} clusters  |  "
    f"silhouette={sil:.3f}  Davies-Bouldin={dbi:.3f}\n"
    f"bruit (outliers) : {n_noise_hdb} CVE ({n_noise_hdb/len(labels_hdb):.1%})",
    fontsize=11, fontweight="bold", y=1.02
)

CLUSTER_COLORS = plt.cm.tab10

# -- scatter PCA coloré par cluster --
ax_sc = axes2[0]
for cl in sorted(set(labels_hdb)):
    mask = labels_hdb == cl
    color = "#111111" if cl == -1 else CLUSTER_COLORS(cl % 10)
    label = "bruit (outliers)" if cl == -1 else f"cluster {cl}"
    ax_sc.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], s=6, alpha=0.5,
                  linewidths=0, label=label)
ax_sc.set_title("clusters HDBSCAN (PCA 2D)", fontweight="bold")
ax_sc.set_xlabel(f"PC1 ({var[0]:.0%})")
ax_sc.set_ylabel(f"PC2 ({var[1]:.0%})")
ax_sc.legend(fontsize=7, markerscale=2)
ax_sc.set_facecolor("#fafafa")
ax_sc.spines[["top", "right"]].set_visible(False)

# -- profil radar simplifié (barres groupées) --
ax_pr = axes2[1]
feat_labels = ["CVSS", "log(EPSS)", "percentile\nEPSS", "nb docs", "sévérité"]
x = np.arange(len(feat_labels))
width = 0.8 / max(n_clusters_hdb, 1)
for i, (cl, row) in enumerate(profils.iterrows()):
    vals = row.values
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    ax_pr.bar(x + i * width, vals_norm, width=width,
              color=CLUSTER_COLORS(cl % 10), alpha=0.8, label=f"cluster {cl}")
ax_pr.set_xticks(x + width * (n_clusters_hdb - 1) / 2)
ax_pr.set_xticklabels(feat_labels, fontsize=8)
ax_pr.set_title("profil normalisé par cluster", fontweight="bold")
ax_pr.set_ylabel("valeur normalisée [0-1]")
ax_pr.legend(fontsize=7)
ax_pr.set_facecolor("#fafafa")
ax_pr.spines[["top", "right"]].set_visible(False)

# -- taille des clusters --
ax_sz = axes2[2]
cluster_ids   = sorted(set(labels_hdb))
cluster_sizes = [int((labels_hdb == cl).sum()) for cl in cluster_ids]
cluster_names = ["bruit" if cl == -1 else f"cluster {cl}" for cl in cluster_ids]
colors_sz     = ["#111111" if cl == -1 else CLUSTER_COLORS(cl % 10) for cl in cluster_ids]
bars = ax_sz.bar(cluster_names, cluster_sizes, color=colors_sz, alpha=0.85)
ax_sz.bar_label(bars, padding=3, fontsize=9)
ax_sz.set_title("taille des clusters", fontweight="bold")
ax_sz.set_ylabel("nb de CVE")
ax_sz.set_facecolor("#fafafa")
ax_sz.spines[["top", "right"]].set_visible(False)
ax_sz.tick_params(axis="x", labelsize=8)

plt.tight_layout()
out2 = "output/clustering_hdbscan.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"sauvegardé : {out2}")
