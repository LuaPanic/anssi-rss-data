"""
classification supervisée — prédire la sévérité des CVE ANSSI
features : - numériques : log(epss), epss_percentile, nb_documents,
                          exploitation, has_cwe, is_alerte, annee_cve
           - risques    : multi-hot des catégories de risque ANSSI (~15)
           - cwe         : one-hot du top-20 des identifiants CWE (+ autre/aucun)
cible    : severite (Faible / Moyenne / Élevée / Critique)
éval     : validation croisée stratifiée (cross_val_predict), sans CVSS
sortie   : output/supervised.png
"""

import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score, classification_report,
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

# exploitation : none=0, poc=1, active=2 — NaN = non confirmé → 0
exploit_map = {"none": 0, "poc": 1, "PoC": 1, "active": 2, "Active": 2}
df["exploit_num"] = df["exploitation"].map(exploit_map).fillna(0)

df["has_cwe"]   = df["cwe"].notna().astype(int)
df["is_alerte"] = df["type"].str.contains("Alerte", na=False).astype(int)
df["annee_cve"] = pd.to_datetime(df["date_cve"], errors="coerce").dt.year
df["annee_cve"] = df["annee_cve"].fillna(df["annee_cve"].median()).astype(int)

# ── features numériques (sans CVSS : la sévérité en est une transformation directe)
NUM_FEATURES = ["log_epss", "epss_percentile", "nb_documents",
                "exploit_num", "has_cwe", "is_alerte", "annee_cve"]

# ── features « risques » : le champ ANSSI est un vocabulaire contrôlé de
#    catégories séparées par « | ». on encode chacune en multi-hot (appartenance
#    exacte). c'est le signal le plus aligné avec l'impact mesuré par le CVSS.
df["risk_set"] = df["risques"].fillna("").apply(
    lambda v: {p.strip() for p in v.split(" | ") if p.strip()}
)
risk_counts = pd.Series(
    [cat for s in df["risk_set"] for cat in s]
).value_counts()
RISK_CATS = risk_counts[risk_counts >= 100].index.tolist()   # ~15 catégories stables
for i, cat in enumerate(RISK_CATS):
    df[f"risk_{i}"] = df["risk_set"].apply(lambda s, c=cat: int(c in s))
RISK_FEATURES = [f"risk_{i}" for i in range(len(RISK_CATS))]

# ── features « cwe » : type de faiblesse (top-20 + autre + aucun) en one-hot
df["cwe_id"]  = df["cwe"].str.extract(r"(CWE-\d+)")[0]
top_cwe       = df["cwe_id"].value_counts().head(20).index
df["cwe_cat"] = df["cwe_id"].where(df["cwe_id"].isin(top_cwe), other="autre")
df["cwe_cat"] = df["cwe_cat"].fillna("aucun")
cwe_dummies   = pd.get_dummies(df["cwe_cat"], prefix="cwe").astype(int)
CWE_FEATURES  = cwe_dummies.columns.tolist()
df = pd.concat([df, cwe_dummies], axis=1)

FEATURES = NUM_FEATURES + RISK_FEATURES + CWE_FEATURES
# noms lisibles pour le panneau d'importance
FEAT_LABELS = (
    ["log(EPSS)", "percentile EPSS", "nb_documents",
     "exploitation", "has_cwe", "is_alerte", "annee_cve"]
    + [c[:38] for c in RISK_CATS]
    + CWE_FEATURES
)

df_clean = df[FEATURES + ["severite"]].dropna()

print(f"  {len(df_clean)} CVE avec données complètes")
print(f"  {len(FEATURES)} features : "
      f"{len(NUM_FEATURES)} num · {len(RISK_FEATURES)} risques · {len(CWE_FEATURES)} cwe")
print("  distribution des classes :")
for s in SEV_ORDER:
    n = (df_clean["severite"] == s).sum()
    print(f"    {s:12s}: {n:5d}  ({n / len(df_clean):.1%})")

X = df_clean[FEATURES].values
y = df_clean["severite"].values

# validation croisée stratifiée : les scores et matrices proviennent des
# prédictions out-of-fold (chaque échantillon prédit par un modèle ne l'ayant
# pas vu). le scaling est dans un Pipeline → ré-ajusté par fold, pas de fuite.
N_SPLITS = 4
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# ───────────────────────────────────────────────────────────────────────────
# classifieurs
# ───────────────────────────────────────────────────────────────────────────
classifiers = [
    ("Régression\nLogistique",    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ("KNN\n(k=7)",                KNeighborsClassifier(n_neighbors=7)),
    ("SVM\n(RBF)",                SVC(kernel="rbf", random_state=42, class_weight="balanced", cache_size=500)),
    ("Arbre de\nDécision",        DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced")),
    ("Random\nForest",            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")),
    ("Gradient\nBoosting",        GradientBoostingClassifier(n_estimators=150, random_state=42)),
    ("Extra\nTrees",              ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")),
    ("AdaBoost",                  AdaBoostClassifier(n_estimators=150, random_state=42)),
    ("Naive\nBayes",              GaussianNB()),
    ("MLP\n(64-32)",              MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
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
    f"  (n={len(df_clean)}, {N_SPLITS}-fold CV)\n"
    f"features : {len(NUM_FEATURES)} numériques + {len(RISK_FEATURES)} risques (multi-hot) "
    f"+ {len(CWE_FEATURES)} CWE (one-hot)   [CVSS exclu]",
    fontsize=10, fontweight="bold", y=1.01,
)

results   = []
cm_axes   = [axes_grid.flat[i] for i in range(len(classifiers))]
recap_ax  = axes_grid.flat[len(classifiers)]      # 11e cellule : barres
feat_ax   = axes_grid.flat[len(classifiers) + 1]  # 12e cellule : importance RF

print(f"\nvalidation croisée de {len(classifiers)} classifieurs ({N_SPLITS} folds)...")
best = {"f1": -1.0, "name": None, "y_pred": None}

for i, (name, clf) in enumerate(tqdm(classifiers, desc="classification", unit="clf")):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    t0     = time.time()
    y_pred = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)
    elapsed = time.time() - t0

    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="macro", zero_division=0)
    results.append({"name": name.replace("\n", " "), "accuracy": acc, "f1_macro": f1, "time": elapsed})

    if f1 > best["f1"]:
        best = {"f1": f1, "name": name.replace("\n", " "), "y_pred": y_pred}

    ax = cm_axes[i]
    ax.set_visible(True)
    ax.set_facecolor("#fafafa")
    ax.spines[["top", "right"]].set_visible(False)

    cm   = confusion_matrix(y, y_pred, labels=SEV_ORDER)
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

# ── panneau 12 : importance des features (Random Forest, top 15) ──────────
feat_ax.set_visible(True)
feat_ax.set_facecolor("#fafafa")
feat_ax.spines[["top", "right"]].set_visible(False)

rf_imp = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
).fit(X, y)
importances = rf_imp.feature_importances_
order = np.argsort(importances)[-15:]          # top 15 croissant
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(order)))
feat_ax.barh(np.arange(len(order)), importances[order], color=colors, alpha=0.9)
feat_ax.set_yticks(np.arange(len(order)))
feat_ax.set_yticklabels([FEAT_LABELS[i] for i in order], fontsize=6)
feat_ax.set_xlabel("Importance (Gini)", fontsize=8)
feat_ax.set_title("Importance des features — top 15\n(Random Forest)", fontsize=8, fontweight="bold")
feat_ax.tick_params(labelsize=6)
for j, imp in enumerate(importances[order]):
    feat_ax.text(imp + 0.002, j, f"{imp:.3f}", va="center", fontsize=6)

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

# rapport par classe du meilleur modèle — révèle le rappel sur la classe rare (Faible)
print(f"\nrapport détaillé par classe — meilleur modèle : {best['name']}")
print(classification_report(y, best["y_pred"], labels=SEV_ORDER, zero_division=0))
