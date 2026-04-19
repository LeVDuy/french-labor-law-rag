"""
Récupération des Conventions Collectives via l'API Légifrance.
Télécharge et structure les textes en Markdown.

Lancer avec :
    python -m app.utils.legifrance_fetcher
"""

import time

import requests
from bs4 import BeautifulSoup

from app.core.config import settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

OAUTH_URL = "https://oauth.piste.gouv.fr/api/oauth/token"
API_BASE_URL = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app"

IDCC_LIST = {
    "1596": "Batiment (Construction)",
    "1486": "Syntec (IT, Consultant)",
    "1979": "HCR (Restaurant, Hotel)",
    "1501": "Restauration rapide (Fast-food)",
    "1090": "Services automobile (Garage)",
}


def get_access_token() -> str:
    payload = {
        "grant_type": "client_credentials",
        "client_id": settings.LEGIFRANCE_CLIENT_ID,
        "client_secret": settings.LEGIFRANCE_CLIENT_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(OAUTH_URL, data=payload, headers=headers)
    response.raise_for_status()
    return response.json().get("access_token")


def fetch_ccn_data(idcc: str, token: str) -> dict:
    endpoint = f"{API_BASE_URL}/consult/kaliContIdcc"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"id": str(idcc)}

    response = requests.post(endpoint, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    logger.error(f"Erreur lors de la récupération IDCC {idcc} : {response.status_code}")
    return None


def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator="\n\n").strip()


def parse_json_to_markdown(node: dict, depth: int, md_lines: list) -> None:
    if "sections" in node and isinstance(node["sections"], list):
        node["sections"].sort(key=lambda x: x.get("intOrdre", 0))

    if "title" in node and node["title"]:
        capped_depth = min(depth + 1, 4)
        prefix = "#" * capped_depth
        md_lines.append(f"\n{prefix} {node['title'].strip()}\n")

    if "articles" in node:
        node["articles"].sort(key=lambda x: x.get("intOrdre", 0))
        for article in node["articles"]:
            if article.get("etat") in ["VIGUEUR", "VIGUEUR_ETEN"]:
                art_num = article.get("num", "Non numéroté")
                content = clean_html(article.get("content", ""))
                md_lines.extend([f"##### Article {art_num}", f"{content}\n"])

    if "sections" in node:
        for child_section in node["sections"]:
            parse_json_to_markdown(child_section, depth + 1, md_lines)


def run_fetcher() -> None:
    output_dir = settings.DATA_PROCESSED_DIR / "conventions"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Authentification API Légifrance...")
        token = get_access_token()
    except requests.exceptions.RequestException as e:
        logger.error(f"Échec de l'authentification : {e}")
        return

    for idcc, name in IDCC_LIST.items():
        logger.info(f"Traitement IDCC {idcc} ({name})...")
        data = fetch_ccn_data(idcc, token)

        if not data:
            continue

        md_lines = [
            f"# Convention collective nationale: {name} (IDCC {idcc})",
            f"Le document decrit les conventions collectives du secteur {name}.\n",
        ]
        root_node = data.get("text", data)
        parse_json_to_markdown(root_node, 1, md_lines)

        filename = f"CCN_{idcc}_{name}.md"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Fichier sauvegardé : {filename}")
        time.sleep(2)

    logger.info("Récupération Légifrance terminée.")


if __name__ == "__main__":
    run_fetcher()
