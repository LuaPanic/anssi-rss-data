"""
génération d'alertes CVE critiques + envoi via Resend
sortie : output/alerts/alert_<date>.html
"""

import os
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
ALERT_FROM     = os.getenv("ALERT_FROM")
ALERT_TO       = os.getenv("ALERT_TO")

OUTPUT_DIR = Path("output/alerts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# critères d'alerte
# -----------------------------------------------------------------------
CVSS_MIN  = 9.0   # critique
EPSS_MIN  = 0.5   # 50% de probabilité d'exploitation

def load_cves() -> pd.DataFrame:
    df = pd.read_csv("output/cve_enriched.csv")
    df["cvss"] = pd.to_numeric(df["cvss"], errors="coerce")
    df["epss"] = pd.to_numeric(df["epss"], errors="coerce")
    return df

def filter_alerts(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        ((df["cvss"] >= CVSS_MIN) | (df["severite"] == "Critique")) &
        (df["epss"] >= EPSS_MIN)
    )
    return df[mask].sort_values("epss", ascending=False).head(20)

# -----------------------------------------------------------------------
# génération HTML
# -----------------------------------------------------------------------
SEV_COLORS = {
    "Critique": "#d62728",
    "Élevée":   "#ff7f0e",
    "Moyenne":  "#f7c948",
    "Faible":   "#2ca02c",
    "Inconnue": "#aaaaaa",
}

def render_row(row) -> str:
    sev   = row.get("severite", "Inconnue")
    color = SEV_COLORS.get(sev, "#aaaaaa")
    cvss  = f"{row['cvss']:.1f}" if pd.notna(row.get("cvss")) else "—"
    epss  = f"{row['epss']:.3f}" if pd.notna(row.get("epss")) else "—"
    liens = " ".join(
        f'<a href="{u.strip()}" style="color:#4c78a8">lien</a>'
        for u in str(row.get("lien", "")).split("|") if u.strip()
    )
    return f"""
    <tr>
      <td style="padding:8px;font-weight:bold;color:{color}">{row['cve_id']}</td>
      <td style="padding:8px">{cvss}</td>
      <td style="padding:8px;color:{color}">{sev}</td>
      <td style="padding:8px">{epss}</td>
      <td style="padding:8px;font-size:12px">{row.get('editeur','—')}</td>
      <td style="padding:8px;font-size:11px;max-width:300px">{str(row.get('description','—'))[:200]}…</td>
      <td style="padding:8px">{liens}</td>
    </tr>"""

def build_html(alerts: pd.DataFrame, today: str) -> str:
    rows = "".join(render_row(row) for _, row in alerts.iterrows())
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>alerte CVE critique — {today}</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#f5f5f5; margin:0; padding:20px; }}
    .container {{ max-width:1100px; margin:auto; background:#fff; border-radius:8px;
                  box-shadow:0 2px 8px rgba(0,0,0,.1); padding:30px; }}
    h1 {{ color:#d62728; font-size:22px; margin-bottom:4px; }}
    .subtitle {{ color:#555; font-size:13px; margin-bottom:24px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th {{ background:#222; color:#fff; padding:10px 8px; text-align:left; }}
    tr:nth-child(even) {{ background:#f9f9f9; }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:4px;
              font-size:11px; font-weight:bold; color:#fff; }}
    .footer {{ margin-top:24px; font-size:11px; color:#aaa; }}
  </style>
</head>
<body>
<div class="container">
  <h1>alertes CVE critiques — ANSSI CERTFR</h1>
  <p class="subtitle">
    date : {today} &nbsp;|&nbsp;
    critères : CVSS ≥ {CVSS_MIN} ou sévérité Critique &nbsp;+&nbsp; EPSS ≥ {EPSS_MIN} &nbsp;|&nbsp;
    {len(alerts)} CVE concernées
  </p>
  <table>
    <thead>
      <tr>
        <th>CVE</th><th>CVSS</th><th>sévérité</th><th>EPSS</th>
        <th>éditeur</th><th>description</th><th>liens ANSSI</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="footer">
    généré automatiquement depuis les données ANSSI CERTFR enrichies via MITRE et FIRST.
  </div>
</div>
</body>
</html>"""

# -----------------------------------------------------------------------
# envoi via Resend
# -----------------------------------------------------------------------
def send_email(subject: str, html: str) -> bool:
    if not RESEND_API_KEY or RESEND_API_KEY.startswith("re_xxx"):
        print("[warn] clé Resend non configurée — envoi ignoré.")
        return False
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "from":    ALERT_FROM,
            "to":      [ALERT_TO],
            "subject": subject,
            "html":    html,
        },
        timeout=15,
    )
    if resp.ok:
        print(f"email envoyé → {ALERT_TO}  (id: {resp.json().get('id')})")
        return True
    print(f"[warn] échec envoi : {resp.status_code} {resp.text}")
    return False

# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------
def main():
    today = date.today().isoformat()

    print("chargement des CVE...")
    df = load_cves()

    print("filtrage des alertes critiques...")
    alerts = filter_alerts(df)
    print(f"  {len(alerts)} CVE retenues (CVSS ≥ {CVSS_MIN} + EPSS ≥ {EPSS_MIN})")

    if alerts.empty:
        print("aucune alerte à envoyer.")
        return

    html = build_html(alerts, today)

    # sauvegarde locale
    out_path = OUTPUT_DIR / f"alert_{today}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"alerte sauvegardée : {out_path}")

    # envoi
    subject = f"[ANSSI] {len(alerts)} CVE critiques à traiter — {today}"
    send_email(subject, html)

if __name__ == "__main__":
    main()
