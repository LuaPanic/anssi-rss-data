"""
visualisations complémentaires — heatmap, boxplot, cumulative, CWE
sortie : output/visualize_extra.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

df = pd.read_csv("output/cve_enriched.csv")
df["cvss"] = pd.to_numeric(df["cvss"], errors="coerce")
df["epss"] = pd.to_numeric(df["epss"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#f5f5f5")
fig.suptitle("analyse CVE ANSSI CERTFR — visualisations complémentaires",
             fontsize=14, fontweight="bold")

BLUE = "#4c78a8"

# ── heatmap corrélation ──────────────────────────────────────────────────
ax = axes[0, 0]
num_cols = ["cvss", "epss", "epss_percentile", "nb_documents"]
labels   = ["CVSS", "EPSS", "percentile\nEPSS", "nb docs"]
corr = df[num_cols].dropna().corr()
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=10, color="white" if abs(corr.values[i, j]) > 0.5 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("heatmap de corrélation", fontweight="bold")
ax.set_facecolor("#fafafa")

# ── boxplot CVSS par éditeur (top 8) ────────────────────────────────────
ax = axes[0, 1]
editeur_counts: Counter = Counter()
for cell in df["editeur"].dropna():
    for e in cell.split(","):
        e = e.strip()
        if e:
            editeur_counts[e] += 1
top8 = [e for e, _ in editeur_counts.most_common(8)]

data_box = []
for e in top8:
    mask = df["editeur"].str.contains(e, na=False, regex=False)
    data_box.append(df.loc[mask, "cvss"].dropna().values)

bp = ax.boxplot(data_box, patch_artist=True, vert=True,
                medianprops={"color": "#d62728", "linewidth": 2})
for patch in bp["boxes"]:
    patch.set_facecolor(BLUE)
    patch.set_alpha(0.7)
ax.set_xticks(range(1, len(top8) + 1))
ax.set_xticklabels(top8, rotation=30, ha="right", fontsize=8)
ax.set_title("dispersion CVSS par éditeur (top 8)", fontweight="bold")
ax.set_ylabel("score CVSS")
ax.set_facecolor("#fafafa")
ax.spines[["top", "right"]].set_visible(False)

# ── courbe cumulative CVE dans le temps ──────────────────────────────────
ax = axes[1, 0]
df_dated = df.dropna(subset=["date"]).sort_values("date")
df_dated["cumul"] = range(1, len(df_dated) + 1)
ax.plot(df_dated["date"], df_dated["cumul"], color=BLUE, linewidth=1.5)
ax.fill_between(df_dated["date"], df_dated["cumul"], alpha=0.15, color=BLUE)
ax.set_title("courbe cumulative des CVE dans le temps", fontweight="bold")
ax.set_xlabel("date")
ax.set_ylabel("nb cumulé de CVE")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_facecolor("#fafafa")
ax.spines[["top", "right"]].set_visible(False)

# ── camembert top CWE ────────────────────────────────────────────────────
ax = axes[1, 1]
cwe_counts: Counter = Counter()
for cell in df["cwe"].dropna():
    cwe = cell.split("(")[0].strip()
    if cwe:
        cwe_counts[cwe] += 1
top_cwe = cwe_counts.most_common(6)
labels_cwe = [c for c, _ in top_cwe] + ["autres"]
vals_cwe   = [v for _, v in top_cwe] + [sum(v for _, v in cwe_counts.most_common()[6:])]
colors_cwe = plt.cm.tab10(np.linspace(0, 1, len(labels_cwe)))
wedges, texts, autotexts = ax.pie(
    vals_cwe, labels=labels_cwe, colors=colors_cwe,
    autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
    startangle=90, pctdistance=0.78,
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    textprops={"fontsize": 7},
)
ax.set_title("top 6 types CWE", fontweight="bold")

plt.tight_layout()
plt.savefig("output/visualize_extra.png", dpi=150, bbox_inches="tight")
print("sauvegardé : output/visualize_extra.png")
