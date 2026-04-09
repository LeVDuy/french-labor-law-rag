import os
import time
import requests
from bs4 import BeautifulSoup

# CONFIGURATION
CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = "your_client_secret_here"
OAUTH_URL = "https://oauth.piste.gouv.fr/api/oauth/token"
API_BASE_URL = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app"

IDCC_LIST = {
    "1596": "Batiment (Construction)",
    "1486": "Syntec (IT, Consultant)",
    "1979": "HCR (Restaurant, Hotel)",
    "1501": "Restauration rapide (Fast-food)",
    "1090": "Services automobile (Garage)"
}

OUTPUT_DIR = "data/markdown_files/conventions_collectives"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API FUNCTIONS
def get_access_token():
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(OAUTH_URL, data=payload, headers=headers)
    response.raise_for_status()
    return response.json().get("access_token")

def fetch_ccn_data(idcc, token):
    endpoint = f"{API_BASE_URL}/consult/kaliContIdcc"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"id": str(idcc)}
    
    response = requests.post(endpoint, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    print(f"Error fetching IDCC {idcc}: {response.status_code}")
    return None

# PARSING FUNCTIONS
def clean_html(raw_html):
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator="\n\n").strip()

def parse_json_to_markdown(node, depth, md_lines):
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

# MAIN EXECUTION
def main():
    try:
        print("Authenticating...")
        token = get_access_token()
    except requests.exceptions.RequestException as e:
        print(f"Authentication failed: {e}")
        return

    for idcc, name in IDCC_LIST.items():
        print(f"Processing IDCC {idcc} ({name})...")
        data = fetch_ccn_data(idcc, token) 

        if not data:
            continue
     
        md_lines = [
            f"# Convention collective nationale: {name} (IDCC {idcc})",
            f"Le document decrit les conventions collectives du secteur {name}.\n"
        ]
        root_node = data.get("text", data) 
        parse_json_to_markdown(root_node, 1, md_lines)
        filename = f"CCN_{idcc}_{name}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
            
        print(f"Saved: {filename}")
        time.sleep(2)

if __name__ == "__main__":
    main()