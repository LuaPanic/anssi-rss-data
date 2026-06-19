# analyse des avis et alertes ANSSI — enrichissement CVE

**SUPDEVINCI 2026 — TD final noté**

Pipeline de veille cyber automatisée sur les bulletins CERTFR de l'ANSSI : extraction, enrichissement CVE, visualisations, clustering, classification supervisée et alertes email.

---

## structure du projet

```
.
├── data/
│   ├── Avis/          # bulletins CERTFR avis (JSON)
│   └── alertes/       # bulletins CERTFR alertes (JSON)
├── output/
│   ├── cve_enriched.csv          # dataset principal (1 ligne / CVE unique)
│   ├── cve_detail.csv            # associations CVE ↔ bulletins
│   ├── clustering.png            # comparatif 11 algorithmes
│   ├── clustering_hdbscan.png    # analyse HDBSCAN détaillée
│   ├── supervised.png            # comparatif 10 classifieurs
│   ├── supervised_validation.png # courbes de validation RF
│   ├── visualize_extra.png       # heatmap, boxplot, cumulative, CWE
│   ├── rf_model.pkl              # modèle Random Forest sauvegardé
│   └── alerts/                   # emails HTML générés
├── enrich_cve.py       # extraction + enrichissement MITRE/EPSS
├── visualize_extra.py  # visualisations exploratoires
├── find_k.py           # méthode du coude + silhouette
├── clustering.py       # comparatif clustering + analyse HDBSCAN
├── supervised.py       # classification supervisée
├── alerts.py           # génération alertes + envoi email Resend
├── anssi_cve_analysis.ipynb  # notebook complet
├── anssi_cve_analysis.html   # export HTML du notebook
├── requirements.txt
└── .env                # credentials (non versionné)
```

---

## installation

```bash
pip install -r requirements.txt
pip install scikit-learn hdbscan matplotlib seaborn python-dotenv nbformat jupyter
```

---

## configuration

Créer un fichier `.env` à la racine :

```env
RESEND_API_KEY=re_xxxxxxxxxxxx
ALERT_FROM=alerts@votre-domaine.io
ALERT_TO=votre@email.com
```

---

## utilisation

### pipeline complet (ordre recommandé)

```bash
# 1. extraction + enrichissement (~37 000 CVE, peut prendre plusieurs minutes)
python3 enrich_cve.py

# 2. visualisations exploratoires
python3 visualize_extra.py

# 3. trouver le k optimal (KMeans)
python3 find_k.py

# 4. comparatif clustering + analyse HDBSCAN
python3 clustering.py

# 5. classification supervisée (10 modèles)
python3 supervised.py

# 6. génération d'alertes + envoi email
python3 alerts.py
```

### notebook interactif

```bash
jupyter notebook anssi_cve_analysis.ipynb
```

---

## données sources

| source | description |
|--------|-------------|
| fichiers JSON locaux (`data/`) | bulletins CERTFR avis et alertes |
| [MITRE CVE API](https://cveawg.mitre.org) | description, CVSS v3, CWE par CVE |
| [FIRST EPSS API](https://api.first.org/data/v1/epss) | probabilité d'exploitation sur 30 jours |

---

## résultats

| étape | résultat |
|-------|----------|
| extraction | ~4 000 avis · 78 alertes · 125 000 associations |
| dataset | **37 279 CVE uniques** enrichies |
| clustering | HDBSCAN : **4 clusters** + détection d'outliers (silhouette=0.40) |
| classification | Random Forest : **F1 macro ~0.5** (sans CVSS dans les features) |
| alertes | CVE avec CVSS ≥ 9.0 ET EPSS ≥ 0.5, envoyées par email |

---

## choix techniques

**pourquoi HDBSCAN pour le clustering ?**
Contrairement à KMeans qui force chaque point dans un cluster, HDBSCAN détecte la structure réelle des données et isole les outliers — CVE avec des combinaisons de scores inhabituelles à surveiller en priorité.

**pourquoi exclure le CVSS du supervisé ?**
Simuler une prédiction précoce : estimer la criticité d'une CVE à partir de signaux indirects (EPSS, CWE, catégories de risque ANSSI) avant d'avoir son score officiel.

**pourquoi la validation croisée stratifiée ?**
`cross_val_predict` avec 4 folds garantit que chaque CVE est prédite par un modèle qui ne l'a pas vu à l'entraînement — évite le data leakage et donne des métriques réalistes.
