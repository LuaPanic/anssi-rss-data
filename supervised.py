"""
classification supervisée — prédire la sévérité des CVE ANSSI
features : log(epss), epss_percentile, nb_documents, exploitation,
           has_cwe, is_alerte, annee_cve  (sans CVSS)
cible    : severite (Faible / Moyenne / Élevée / Critique)
sortie   : output/supervised.png
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
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

# features sans CVSS (la sévérité en est une transformation directe)
FEATURES = ["log_epss", "epss_percentile", "nb_documents",
            "exploit_num", "has_cwe", "is_alerte", "annee_cve"]
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
    ("Régression\nLogistique",    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ("KNN\n(k=7)",                KNeighborsClassifier(n_neighbors=7)),
    ("SVM\n(RBF)",                SVC(kernel="rbf", random_state=42, class_weight="balanced")),
    ("Arbre de\nDécision",        DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced")),
    ("Random\nForest",            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")),
    ("Gradient\nBoosting",        GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("Extra\nTrees",              ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")),
    ("AdaBoost",                  AdaBoostClassifier(n_estimators=100, random_state=42)),
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
    f"  (n={len(df_clean)}, test=25%)\n"
    f"features : log(EPSS) · percentile EPSS · nb_documents · exploitation · has_cwe · is_alerte · annee_cve   [CVSS exclu]",
    fontsize=10, fontweight="bold", y=1.01,
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

    feat_names  = ["log(EPSS)", "percentile\nEPSS", "nb_\ndocuments",
                   "exploitation", "has_cwe", "is_alerte", "annee_cve"]
    importances = rf_clf.feature_importances_
    idx         = np.argsort(importances)
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(idx)))
    feat_ax.barh(np.arange(len(idx)), importances[idx], color=colors, alpha=0.85)
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

# ── sauvegarde du meilleur modèle (Random Forest) ────────────────────────
print("\nvalidation approfondie — Random Forest (meilleur modèle)...")
rf_best = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_best.fit(X_train_s, y_train)
y_pred_rf = rf_best.predict(X_test_s)

print("\nrapport de classification :")
print(classification_report(y_test, y_pred_rf, target_names=SEV_ORDER, zero_division=0))

cv_scores = cross_val_score(rf_best, scaler.transform(X), y, cv=5, scoring="f1_macro", n_jobs=-1)
print(f"cross-validation 5-fold F1 macro : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

joblib.dump({"model": rf_best, "scaler": scaler, "features": FEATURES}, "output/rf_model.pkl")
print("modèle sauvegardé : output/rf_model.pkl")

# ── figure validation approfondie ────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), facecolor="#f5f5f5")
fig2.suptitle("validation Random Forest — sévérité CVE ANSSI", fontsize=12, fontweight="bold")

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    scaler.transform(X), y, cv=5, scoring="f1_macro",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
)
ax_lc = axes2[0]
ax_lc.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#4c72b0", label="train")
ax_lc.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#4c72b0")
ax_lc.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#dd8452", label="validation")
ax_lc.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.15, color="#dd8452")
ax_lc.set_title("courbe d'apprentissage (F1 macro)", fontweight="bold")
ax_lc.set_xlabel("taille du jeu d'entraînement")
ax_lc.set_ylabel("F1 macro")
ax_lc.legend(fontsize=9)
ax_lc.set_facecolor("#fafafa")
ax_lc.spines[["top", "right"]].set_visible(False)

# distribution des prédictions vs réalité
ax_dist = axes2[1]
sev_colors = ["#2ca02c", "#f7c948", "#ff7f0e", "#d62728"]
x_pos = np.arange(len(SEV_ORDER))
width = 0.35
real_counts = [int((y_test == s).sum()) for s in SEV_ORDER]
pred_counts = [int((y_pred_rf == s).sum()) for s in SEV_ORDER]
ax_dist.bar(x_pos - width/2, real_counts, width, label="réel",    color=sev_colors, alpha=0.6)
ax_dist.bar(x_pos + width/2, pred_counts, width, label="prédit",  color=sev_colors, alpha=1.0)
ax_dist.set_xticks(x_pos)
ax_dist.set_xticklabels(SEV_ORDER, fontsize=9)
ax_dist.set_title("distribution réel vs prédit (test)", fontweight="bold")
ax_dist.set_ylabel("nb de CVE")
ax_dist.legend(fontsize=9)
ax_dist.set_facecolor("#fafafa")
ax_dist.spines[["top", "right"]].set_visible(False)

# scores CV par fold
ax_cv = axes2[2]
ax_cv.bar(np.arange(1, 6), cv_scores, color="#4c72b0", alpha=0.8)
ax_cv.axhline(cv_scores.mean(), color="#d62728", linestyle="--", linewidth=1.5,
              label=f"moyenne : {cv_scores.mean():.3f}")
ax_cv.set_title("F1 macro par fold (cross-validation 5-fold)", fontweight="bold")
ax_cv.set_xlabel("fold")
ax_cv.set_ylabel("F1 macro")
ax_cv.set_xticks(np.arange(1, 6))
ax_cv.legend(fontsize=9)
ax_cv.set_ylim(0, 1)
ax_cv.set_facecolor("#fafafa")
ax_cv.spines[["top", "right"]].set_visible(False)
for i, v in enumerate(cv_scores):
    ax_cv.text(i + 1, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

plt.tight_layout()
out2 = "output/supervised_validation.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"sauvegardé : {out2}")
