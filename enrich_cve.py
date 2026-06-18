"""
enrichissement des CVE à partir des alertes et avis CERTFR (ANSSI).

sources CERTFR : data/alertes/  +  data/Avis/   (100% local)
enrichissement : data/mitre/  +  data/first/    (local d'abord, API en repli)
sortie         : output/cve_enriched.json/.csv  +  output/cve_detail.json/.csv

chaque ligne du détail représente une association CVE ↔ document CERTFR. le CERTFR
fournit le contexte (risques, systèmes affectés, éditeur...), MITRE et EPSS fournissent
le score CVSS, le CWE, la description et la probabilité d'exploitation.
un deuxième fichier résumé agrège par CVE unique.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR    = Path("data")
ALERTES_DIR = DATA_DIR / "alertes"
AVIS_DIR    = DATA_DIR / "Avis"
MITRE_DIR   = DATA_DIR / "mitre"
FIRST_DIR   = DATA_DIR / "first"
OUTPUT_DIR  = Path("output")

# si un CVE n'a pas de fichier local, interroger les API en direct.
# garder False = 100% local. à True, garder REQUETES_SIMULTANEES bas (consigne d'accès responsable).
ENRICHIR_EN_LIGNE    = True
REQUETES_SIMULTANEES = 10

CVSS_VERSIONS_PAR_PRIORITE = ("cvssV4_0", "cvssV3_1", "cvssV3_0", "cvssV2_0")


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_certfr_files(directory: Path) -> list[dict]:
    records = []
    if not directory.exists():
        print(f"[warn] répertoire introuvable : {directory}")
        return records
    files = sorted(directory.iterdir())
    for path in tqdm(files, desc=f"  {directory.name}", unit="fichier"):
        if path.is_file():
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as e:
                print(f"[warn] {path.name} : {e}")
    return records


def load_local_json(directory: Path, name: str) -> dict | None:
    path = directory / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Enrichissement MITRE / EPSS
# ---------------------------------------------------------------------------

def chemin(donnees, *cles):
    courant = donnees or {}
    for cle in cles:
        if not isinstance(courant, dict):
            return None
        courant = courant.get(cle)
    return courant


def fetch_mitre(cve_id: str) -> dict | None:
    local = load_local_json(MITRE_DIR, cve_id)
    if local is not None:
        return local
    if not ENRICHIR_EN_LIGNE:
        return None
    try:
        r = requests.get(f"https://cveawg.mitre.org/api/cve/{cve_id}", timeout=15)
        return r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def fetch_epss_local(cve_id: str) -> dict | None:
    local = load_local_json(FIRST_DIR, cve_id)
    if local is None:
        return None
    data = local.get("data", [])
    if not data:
        return {"epss": None, "percentile": None}
    row = data[0]
    pct = row.get("percentile")
    return {"epss": float(row["epss"]), "percentile": float(pct) if pct is not None else None}


def fetch_epss_batch(cves: list[str]) -> dict:
    try:
        r = requests.get(f"https://api.first.org/data/v1/epss?cve={','.join(cves)}", timeout=15)
        rows = r.json().get("data", []) if r.ok else []
        return {row["cve"]: {"epss": float(row["epss"]), "percentile": float(row["percentile"])} for row in rows}
    except (requests.RequestException, ValueError, KeyError):
        return {}


def format_cvss_version(cle: str | None) -> str | None:
    return "v" + cle.replace("cvssV", "").replace("_", ".") if cle else None


def extract_cvss(mitre: dict | None):
    mesures = list(chemin(mitre, "containers", "cna", "metrics") or [])
    for bloc in chemin(mitre, "containers", "adp") or []:
        mesures.extend(bloc.get("metrics") or [])
    for version in CVSS_VERSIONS_PAR_PRIORITE:
        for mesure in mesures:
            cvss = mesure.get(version)
            if cvss and cvss.get("baseScore") is not None:
                return cvss["baseScore"], cvss.get("baseSeverity"), version
    return None, None, None


def extract_cwe(mitre: dict | None) -> str | None:
    types = chemin(mitre, "containers", "cna", "problemTypes") or [{}]
    descriptions = types[0].get("descriptions") or [{}]
    identifiant = descriptions[0].get("cweId")
    libelle = re.sub(r"^CWE-\d+:?\s*", "", descriptions[0].get("description", "") or "").strip()
    if identifiant:
        return f"{identifiant} ({libelle})" if libelle else identifiant
    return libelle or None


def extract_description(mitre: dict | None) -> str | None:
    descriptions = chemin(mitre, "containers", "cna", "descriptions") or []
    anglaise = next((d for d in descriptions if (d.get("lang") or "").startswith("en")), None)
    choisie = anglaise or (descriptions[0] if descriptions else {})
    return choisie.get("value")


def extract_exploitation(mitre: dict | None) -> str | None:
    for bloc in chemin(mitre, "containers", "adp") or []:
        for mesure in bloc.get("metrics") or []:
            autre = mesure.get("other") or {}
            if autre.get("type") == "ssvc":
                for option in chemin(autre, "content", "options") or []:
                    if "Exploitation" in option:
                        return option["Exploitation"]
    return None


def extract_cve_date(mitre: dict | None) -> str | None:
    date = chemin(mitre, "cveMetadata", "datePublished")
    return date[:10] if date else None


def severite_fr(score) -> str:
    if pd.isna(score):
        return "Inconnue"
    if score <= 3:
        return "Faible"
    if score <= 6:
        return "Moyenne"
    if score <= 8:
        return "Élevée"
    return "Critique"


def enrich_cves(cve_ids: list[str]) -> dict[str, dict]:
    unique = list(dict.fromkeys(cve_ids))

    # EPSS : local d'abord, le reste groupé en lots (l'API accepte plusieurs CVE par appel)
    epss = {}
    missing = []
    for cve in unique:
        local = fetch_epss_local(cve)
        if local is not None:
            epss[cve] = local
        else:
            missing.append(cve)
    if ENRICHIR_EN_LIGNE:
        for debut in range(0, len(missing), 100):
            epss.update(fetch_epss_batch(missing[debut:debut + 100]))

    # MITRE : un appel par CVE, parallélisé pour le repli en ligne
    if ENRICHIR_EN_LIGNE:
        with ThreadPoolExecutor(max_workers=REQUETES_SIMULTANEES) as ex:
            mitre_list = list(tqdm(ex.map(fetch_mitre, unique), total=len(unique), desc="  MITRE", unit="cve"))
    else:
        mitre_list = [fetch_mitre(cve) for cve in tqdm(unique, desc="  MITRE", unit="cve")]

    enriched = {}
    for cve, mitre in zip(unique, mitre_list):
        score, severity, version = extract_cvss(mitre)
        epss_info = epss.get(cve) or {}
        enriched[cve] = {
            "cvss":            score,
            "cvss_version":    format_cvss_version(version),
            "base_severity":   severity,
            "cwe":             extract_cwe(mitre),
            "epss":            epss_info.get("epss"),
            "epss_percentile": epss_info.get("percentile"),
            "exploitation":    extract_exploitation(mitre),
            "date_cve":        extract_cve_date(mitre),
            "description":     extract_description(mitre),
        }
    return enriched


# ---------------------------------------------------------------------------
# Extraction des associations CVE → contexte CERTFR
# ---------------------------------------------------------------------------

def extract_rows(records: list[dict], source_type: str) -> list[dict]:
    rows = []
    for rec in records:
        cves = rec.get("cves") or []
        if not cves:
            continue

        revisions = rec.get("revisions") or []
        dates = sorted(r["revision_date"] for r in revisions if r.get("revision_date"))

        risks = " | ".join(
            r.get("description", "") for r in (rec.get("risks") or []) if r.get("description")
        )

        # systèmes affectés — vendor, produit et description de version
        affected_parts = []
        vendors = set()
        products = set()
        for s in rec.get("affected_systems") or []:
            desc = s.get("description", "")
            prod = s.get("product") or {}
            vendor = (prod.get("vendor") or {}).get("name", "")
            product_name = prod.get("name", "")
            if vendor:
                vendors.add(vendor)
            if product_name and product_name.lower() != "n/a":
                products.add(product_name)
            if desc:
                affected_parts.append(desc)

        links = " | ".join(lk.get("url", "") for lk in (rec.get("links") or []) if lk.get("url"))

        for cve_entry in cves:
            cve_id = (cve_entry.get("name") or "").upper().strip()
            if not cve_id:
                continue
            rows.append({
                "id_anssi":           rec.get("reference", ""),
                "titre_anssi":        rec.get("title", ""),
                "type":               source_type.capitalize(),
                "date":               dates[0][:10] if dates else None,
                "cve_id":             cve_id,
                "cvss":               None,
                "cvss_version":       None,
                "base_severity":      None,
                "severite":           None,
                "cwe":                None,
                "epss":               None,
                "epss_percentile":    None,
                "exploitation":       None,
                "date_cve":           None,
                "risques":            risks,
                "lien":               links,
                "description":        None,
                "editeur":            ", ".join(sorted(vendors)),
                "produit":            ", ".join(sorted(products)),
                "versions_affectees": " | ".join(affected_parts),
            })
    return rows


# ---------------------------------------------------------------------------
# Agrégation par CVE unique
# ---------------------------------------------------------------------------

def aggregate_by_cve(df: pd.DataFrame) -> pd.DataFrame:
    """un enregistrement par CVE unique, champs multi-valeurs concaténés."""
    def join_unique(series: pd.Series, sep=" | ") -> str:
        values = []
        for v in series.dropna():
            for part in str(v).split(sep):
                part = part.strip()
                if part and part not in values:
                    values.append(part)
        return sep.join(values)

    agg = df.groupby("cve_id", sort=True).agg(
        id_anssi          =("id_anssi",     lambda s: " | ".join(sorted(set(s)))),
        titre_anssi       =("titre_anssi",  lambda s: join_unique(s, " | ")),
        type              =("type",         lambda s: " | ".join(sorted(set(s)))),
        date              =("date",         "min"),
        cvss              =("cvss",         "first"),
        cvss_version      =("cvss_version", "first"),
        base_severity     =("base_severity", "first"),
        severite          =("severite",     "first"),
        cwe               =("cwe",          lambda s: join_unique(s, " | ")),
        epss              =("epss",         "first"),
        epss_percentile   =("epss_percentile", "first"),
        exploitation      =("exploitation", "first"),
        date_cve          =("date_cve",     "first"),
        risques           =("risques",      lambda s: join_unique(s, " | ")),
        lien              =("lien",         lambda s: join_unique(s, " | ")),
        description       =("description",  "first"),
        editeur           =("editeur",      lambda s: join_unique(s, ", ")),
        produit           =("produit",      lambda s: join_unique(s, ", ")),
        versions_affectees=("versions_affectees", lambda s: join_unique(s, " | ")),
        nb_documents      =("id_anssi",     "count"),
    ).reset_index()

    return agg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("chargement des fichiers CERTFR...")
    alertes = load_certfr_files(ALERTES_DIR)
    avis    = load_certfr_files(AVIS_DIR)
    print(f"  {len(alertes)} alertes  |  {len(avis)} avis")

    print("\nextraction des associations CVE...")
    rows = extract_rows(alertes, "alerte") + extract_rows(avis, "avis")
    print(f"  {len(rows)} associations extraites")

    df = pd.DataFrame(rows)
    df.sort_values(["id_anssi", "cve_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\nenrichissement MITRE / EPSS...")
    enriched = enrich_cves(df["cve_id"].tolist())
    for field in ("cvss", "cvss_version", "base_severity", "cwe", "epss",
                  "epss_percentile", "exploitation", "date_cve", "description"):
        df[field] = df["cve_id"].map(lambda c, f=field: enriched.get(c, {}).get(f))
    df["cvss"]            = pd.to_numeric(df["cvss"], errors="coerce")
    df["epss"]           = pd.to_numeric(df["epss"], errors="coerce")
    df["epss_percentile"] = pd.to_numeric(df["epss_percentile"], errors="coerce")
    df["severite"]       = df["cvss"].map(severite_fr)

    print("\nagrégation par CVE unique...")
    df_agg = aggregate_by_cve(df)
    print(f"  {len(df_agg)} CVE uniques")

    # export — associations détaillées
    detail_json = OUTPUT_DIR / "cve_detail.json"
    detail_csv  = OUTPUT_DIR / "cve_detail.csv"
    df.to_json(detail_json, orient="records", force_ascii=False, indent=2)
    df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    # export — résumé agrégé
    agg_json = OUTPUT_DIR / "cve_enriched.json"
    agg_csv  = OUTPUT_DIR / "cve_enriched.csv"
    df_agg.to_json(agg_json, orient="records", force_ascii=False, indent=2)
    df_agg.to_csv(agg_csv, index=False, encoding="utf-8-sig")

    print("\nrésultats :")
    print(f"  associations CVE-CERTFR : {len(df)}")
    print(f"  CVE uniques             : {len(df_agg)}")
    print(f"  CVE avec score CVSS     : {df_agg['cvss'].notna().sum()}")
    print(f"  CVE avec score EPSS     : {df_agg['epss'].notna().sum()}")
    print(f"  CVE dans alertes        : {df[df['type']=='Alerte']['cve_id'].nunique()}")
    print(f"  CVE dans avis           : {df[df['type']=='Avis']['cve_id'].nunique()}")
    print(f"\n  {detail_csv}")
    print(f"  {agg_csv}")

    print("\naperçu CVE agrégés (top 5 par nb_documents) :")
    cols = ["cve_id", "nb_documents", "cvss", "severite", "epss", "editeur"]
    print(df_agg.nlargest(5, "nb_documents")[cols].to_string(index=False))


if __name__ == "__main__":
    main()