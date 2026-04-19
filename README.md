#  French Labor Law RAG — Assistant Juridique

![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688) 
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange) 
![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_Search-red) 
![Score](https://img.shields.io/badge/Evaluation-80%25-success)

Un système **RAG (Retrieval-Augmented Generation)** performant conçu pour naviguer dans le droit du travail français (Code du Travail, Code de la Sécurité Sociale, Conventions Collectives majeures).

##  Architecture & Stack Tech
- **Recherche Hybride** : Dense (BGE-M3) + Sparse (BM25) via **Qdrant**.
- **Reranking** : Filtrage de haute précision avec CrossEncoder CamemBERT.
- **Parent Document Retrieval** : Recherche sur de petits segments mais injection de l'article de loi complet au LLM pour 0 hallucination.
- **Orchestration Agentique** : Routage d'intention et reformulation contextuelle via **LangGraph**.
- **Génération** : Configuré pour **Llama-3.3-70B (Groq)** pour une vitesse et précision maximales (support local via LM Studio toujours disponible).
- **Précision** : Évalué à **80.00%** sur des benchmarks stricts (LLM-as-a-judge).

##  Installation Rapide (Sans Docker)

```bash
# 1. Cloner et installer les dépendances
git clone https://github.com/LeVDuy/french-labor-law-rag.git
cd french-labor-law-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurer l'environnement
cp .env.example .env
# Important: Remplissez GROQ_API_KEY dans le fichier .env

# 3. Lancer Qdrant (requis)
docker run -d -p 6333:6333 qdrant/qdrant

# 4. Ingest des lois dans la base de données
python -m app.rag.ingest

# 5. Démarrer l'interface web professionnelle
streamlit run ui/streamlit_app.py
```
*(L'interface web sera disponible sur `http://localhost:8501`)*

##  Déploiement Complet (Docker Compose)
Si vous voulez lancer l'API backend et Qdrant en un clic :
```bash
docker compose up --build
```
*(API disponible sur `http://localhost:8000/docs`)*

##  Évaluation
Le système inclut un framework d'évaluation automatisé testant 11 scénarios complexes (pièges légaux, calculs, conditions).
```bash
python -m tests.evaluation
```