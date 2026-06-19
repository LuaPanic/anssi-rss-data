"""
classification supervisée — prédire la sévérité des CVE ANSSI
features : log(epss), epss_percentile, nb_documents  (sans CVSS)
cible    : severite (Faible / Moyenne / Élevée / Critique)
sortie   : output/supervised.png
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────────
# données
# ───────────────────────────────────────────────────────────────────────────
print("chargement des données...")
df = pd.read_csv("output/cve_enriched.csv")
df["cvss"]            = pd.to_numeric(df["cvss"], errors="coerce")
df["epss"]            = pd.to_numeric(df["epss"], errors="coerce")
df["epss_percentile"] = pd.to_numeric(df["epss_percentile"], errors="coerce")
df["nb_documents"]    = pd.to_numeric(df["nb_documents"], errors="coerce")

SEV_ORDER  = ["Faible", "Moyenne", "Élevée", "Critique"]   # ordre croissant
SEV_LABELS = ["F",      "M",       "É",            "C"]          # abréviations pour matrice

df = df[df["severite"].isin(SEV_ORDER)].copy()
df["log_epss"] = np.log1p(df["epss"])

# features sans CVSS (la sévérité en est une transformation directe)
FEATURES = ["log_epss", "epss_percentile", "nb_documents"]
df_clean = df[FEATURES + ["severite"]].dropna()

print(f"  {len(df_clean)} CVE avec données complètes")
print("  distribution des classes :")
for s in SEV_ORDER:
    n = (df_clean["severite"] == s).sum()
    print(f"    {s:12s}: {n:5d}  ({n / len(df_clean):.1%})")

X = df_clean[FEATURES].values
y = df_clean["severite"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ───────────────────────────────────────────────────────────────────────────
# classifieurs
# ───────────────────────────────────────────────────────────────────────────
classifiers = [
    ("Régression\nLogistique",    LogisticRegression(max_iter=1000, random_state=42)),
    ("KNN\n(k=7)",                KNeighborsClassifier(n_neighbors=7)),
    ("SVM\n(RBF)",                SVC(kernel="rbf", random_state=42)),
    ("Arbre de\nDécision",        DecisionTreeClassifier(max_depth=6, random_state=42)),
    ("Random\nForest",            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("Gradient\nBoosting",        GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("Extra\nTrees",              ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("AdaBoost",                  AdaBoostClassifier(n_estimators=100, random_state=42)),
    ("Naive\nBayes",              GaussianNB()),
    ("MLP\n(64-32)",               MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
]

# ───────────────────────────────────────────────────────────────────────────
# figure  (3 × 4 = 12 cellules : 10 matrices + 1 barre récap + 1 importance)
# ───────────────────────────────────────────────────────────────────────────
N_COLS = 4
N_ROWS = 3
fig, axes_grid = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(N_COLS * 4.2, N_ROWS * 4.2),
    facecolor="#f5f5f5",
)

for ax in axes_grid.flat:
    ax.set_visible(False)

fig.suptitle(
    f"classification supervisée — sévérité CVE ANSSI"
    f"  (n={len(df_clean)}, test=25%)\n"
    f"features : log(EPSS) · percentile EPSS · nb_documents   [CVSS exclu]",
    fontsize=11, fontweight="bold", y=1.01,
)

results   = []
cm_axes   = [axes_grid.flat[i] for i in range(len(classifiers))]
recap_ax  = axes_grid.flat[len(classifiers)]      # 11e cellule : barres
feat_ax   = axes_grid.flat[len(classifiers) + 1]  # 12e cellule : importance RF

print(f"\nentraînement de {len(classifiers)} classifieurs...")
rf_clf = None   # récupéré pour l'importance des features

for i, (name, clf) in enumerate(tqdm(classifiers, desc="classification", unit="clf")):
    t0    = time.time()
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    elapsed = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    results.append({"name": name.replace("\n", " "), "accuracy": acc, "f1_macro": f1, "time": elapsed})

    if "Random" in name:
        rf_clf = clf

    ax = cm_axes[i]
    ax.set_visible(True)
    ax.set_facecolor("#fafafa")
    ax.spines[["top", "right"]].set_visible(False)

    cm   = confusion_matrix(y_test, y_pred, labels=SEV_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SEV_LABELS)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    ax.set_title(
        f"{name}\nacc={acc:.2f}  F1={f1:.2f}  ({elapsed:.1f}s)",
        fontsize=7.5, fontweight="bold", pad=4,
    )
    ax.set_xlabel("Prédit", fontsize=7)
    ax.set_ylabel("Réel",   fontsize=7)
    ax.tick_params(labelsize=6)

# ── panneau 11 : barres comparatives ──────────────────────────────────────
res_df = pd.DataFrame(results).sort_values("f1_macro", ascending=True)
recap_ax.set_visible(True)
recap_ax.set_facecolor("#fafafa")
recap_ax.spines[["top", "right"]].set_visible(False)

y_pos = np.arange(len(res_df))
recap_ax.barh(y_pos - 0.2, res_df["f1_macro"],  height=0.4, color="#4c72b0", alpha=0.85, label="F1 macro")
recap_ax.barh(y_pos + 0.2, res_df["accuracy"],  height=0.4, color="#dd8452", alpha=0.85, label="Accuracy")
recap_ax.set_yticks(y_pos)
recap_ax.set_yticklabels(res_df["name"].str.replace(" ", "\n"), fontsize=6.5)
recap_ax.set_xlabel("Score", fontsize=8)
recap_ax.set_title("Comparatif — F1 macro · Accuracy", fontsize=8, fontweight="bold")
recap_ax.legend(fontsize=7, loc="lower right")
recap_ax.set_xlim(0, 1.08)
recap_ax.tick_params(labelsize=7)
for j, (_, row) in enumerate(res_df.iterrows()):
    recap_ax.text(row["f1_macro"] + 0.01, j - 0.2, f"{row['f1_macro']:.2f}",
                  va="center", fontsize=6)

# ── panneau 12 : importance des features (Random Forest) ──────────────────
if rf_clf is not None:
    feat_ax.set_visible(True)
    feat_ax.set_facecolor("#fafafa")
    feat_ax.spines[["top", "right"]].set_visible(False)

    feat_names  = ["log(EPSS)", "percentile\nEPSS", "nb_\ndocuments"]
    importances = rf_clf.feature_importances_
    idx         = np.argsort(importances)
    feat_ax.barh(np.arange(len(idx)), importances[idx],
                 color=["#55a868", "#4c72b0", "#dd8452"][:len(idx)], alpha=0.85)
    feat_ax.set_yticks(np.arange(len(idx)))
    feat_ax.set_yticklabels([feat_names[i] for i in idx], fontsize=8)
    feat_ax.set_xlabel("Importance (Gini)", fontsize=8)
    feat_ax.set_title("Importance des features\n(Random Forest)", fontsize=8, fontweight="bold")
    feat_ax.tick_params(labelsize=7)
    for j, imp in enumerate(importances[idx]):
        feat_ax.text(imp + 0.005, j, f"{imp:.3f}", va="center", fontsize=7)

plt.tight_layout()
out = "output/supervised.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nsauvegardé : {out}")

# ── résumé terminal ──────────────────────────────────────────────────────
res_sorted = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
print("\nclassement par F1 macro :")
print(f"{'classifieur':28s}  {'accuracy':>9s}  {'F1 macro':>9s}  {'temps':>8s}")
print("-" * 62)
for _, row in res_sorted.iterrows():
    print(f"{row['name']:28s}  {row['accuracy']:9.3f}  {row['f1_macro']:9.3f}  {row['time']:7.2f}s")
