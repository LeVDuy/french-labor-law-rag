"""
FastAPI backend — Assistant Juridique en Droit du Travail Français.

Lancer avec :
    uvicorn app.api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Démarrage de l'API RAG — Droit du Travail Français")
    logger.info(f"Qdrant : {settings.QDRANT_URL}")
    logger.info(f"LLM : {settings.LLM_BASE_URL}")
    logger.info("=" * 60)
    yield


app = FastAPI(
    title="French Labor Law RAG API",
    description=(
        "API REST pour un assistant juridique en droit du travail français. "
        "Utilise un pipeline RAG avec recherche hybride (BM25 + Dense) "
        "et reranking CrossEncoder sur une base Qdrant."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from app.schemas.qa import QuestionRequest, AnswerResponse, HealthResponse


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    qdrant_ok = False
    try:
        from app.rag.vectorstore import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
        qdrant_ok = True
    except Exception as e:
        logger.warning(f"Échec du health check Qdrant : {e}")

    return HealthResponse(
        status="ok",
        service="french-labor-law-rag",
        qdrant_connected=qdrant_ok,
    )


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask_question(request: QuestionRequest):
    logger.info(f"Question reçue : {request.question[:80]}...")

    try:
        from app.rag.pipeline import ask_question as pipeline_ask
        response = pipeline_ask(
            question=request.question,
            chat_history=request.chat_history or [],
        )
        return response
    except Exception as e:
        logger.error(f"Erreur pipeline : {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du pipeline RAG: {str(e)}",
        )

