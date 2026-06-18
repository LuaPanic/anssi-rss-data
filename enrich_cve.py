"""
Enrichissement des CVE à partir des alertes et avis CERTFR (ANSSI) — 100% local.

Sources : data/alertes/  +  data/Avis/
Sortie  : output/cve_enriched.json  +  output/cve_enriched.csv

Chaque ligne du fichier de sortie représente une association CVE ↔ document CERTFR,
avec toutes les métadonnées disponibles localement (risques, systèmes affectés, etc.).
Un deuxième fichier résumé agrège par CVE unique.
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data")
ALERTES_DIR = DATA_DIR / "alertes"
AVIS_DIR    = DATA_DIR / "Avis"
OUTPUT_DIR  = Path("output")


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_certfr_files(directory: Path) -> list[dict]:
    records = []
    if not directory.exists():
        print(f"[WARN] Répertoire introuvable : {directory}")
        return records
    files = sorted(directory.iterdir())
    for path in tqdm(files, desc=f"  {directory.name}", unit="fichier"):
        if path.is_file():
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as e:
                print(f"[WARN] {path.name} : {e}")
    return records


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

        # Systèmes affectés — on garde vendor, produit et description de version
        affected_list = rec.get("affected_systems") or []
        affected_parts = []
        vendors = set()
        products = set()
        for s in affected_list:
            desc = s.get("description", "")
            prod = (s.get("product") or {})
            vendor = (prod.get("vendor") or {}).get("name", "")
            product_name = prod.get("name", "")
            if vendor:
                vendors.add(vendor)
            if product_name:
                products.add(product_name)
            if desc:
                affected_parts.append(desc)

        # Liens de documentation
        links = " | ".join(
            lk.get("url", "") for lk in (rec.get("links") or []) if lk.get("url")
        )

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
                "base_severity":      None,
                "cwe":                None,
                "epss":               None,
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
    """Un enregistrement par CVE unique, champs multi-valeurs concaténés."""
    def join_unique(series: pd.Series, sep=" | ") -> str:
        values = []
        for v in series.dropna():
            for part in str(v).split(sep):
                part = part.strip()
                if part and part not in values:
                    values.append(part)
        return sep.join(values)

    agg = df.groupby("cve_id", sort=True).agg(
        id_anssi          =("id_anssi",    lambda s: " | ".join(sorted(set(s)))),
        titre_anssi       =("titre_anssi", lambda s: join_unique(s, " | ")),
        type              =("type",        lambda s: " | ".join(sorted(set(s)))),
        date              =("date",        "min"),
        cvss              =("cvss",        "first"),
        base_severity     =("base_severity", "first"),
        cwe               =("cwe",         lambda s: join_unique(s, " | ")),
        epss              =("epss",        "first"),
        lien              =("lien",        lambda s: join_unique(s, " | ")),
        description       =("description", "first"),
        editeur           =("editeur",     lambda s: join_unique(s, ", ")),
        produit           =("produit",     lambda s: join_unique(s, ", ")),
        versions_affectees=("versions_affectees", lambda s: join_unique(s, " | ")),
        nb_documents      =("id_anssi",    "count"),
    ).reset_index()

    return agg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=== Chargement des fichiers CERTFR ===")
    alertes = load_certfr_files(ALERTES_DIR)
    avis    = load_certfr_files(AVIS_DIR)
    print(f"  {len(alertes)} alertes  |  {len(avis)} avis")

    print("\n=== Extraction des associations CVE ===")
    rows = extract_rows(alertes, "alerte") + extract_rows(avis, "avis")
    print(f"  {len(rows)} associations extraites")

    df = pd.DataFrame(rows)

    df.sort_values(["id_anssi", "cve_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\n=== Agrégation par CVE unique ===")
    df_agg = aggregate_by_cve(df)
    print(f"  {len(df_agg)} CVE uniques")

    # Export — associations détaillées
    detail_json = OUTPUT_DIR / "cve_detail.json"
    detail_csv  = OUTPUT_DIR / "cve_detail.csv"
    df.to_json(detail_json, orient="records", force_ascii=False, indent=2)
    df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    # Export — résumé agrégé
    agg_json = OUTPUT_DIR / "cve_enriched.json"
    agg_csv  = OUTPUT_DIR / "cve_enriched.csv"
    df_agg.to_json(agg_json, orient="records", force_ascii=False, indent=2)
    df_agg.to_csv(agg_csv, index=False, encoding="utf-8-sig")

    print(f"\n=== Résultats ===")
    print(f"  Associations CVE-CERTFR : {len(df)}")
    print(f"  CVE uniques             : {len(df_agg)}")
    print(f"  CVE dans alertes        : {df[df['type']=='Alerte']['cve_id'].nunique()}")
    print(f"  CVE dans avis           : {df[df['type']=='Avis']['cve_id'].nunique()}")
    print(f"\n  {detail_csv}")
    print(f"  {agg_csv}")

    print("\nAperçu CVE agrégés (top 5 par nb_documents) :")
    cols = ["cve_id", "nb_documents", "editeur", "base_severity"]
    print(df_agg.nlargest(5, "nb_documents")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
