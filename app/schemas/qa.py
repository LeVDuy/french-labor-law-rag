"""
Schémas Pydantic pour les requêtes et réponses de l'API RAG.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question juridique en droit du travail français.",
        json_schema_extra={
            "examples": ["Quels sont les droits du salarié en période d'essai ?"]
        },
    )
    chat_history: Optional[List[dict]] = Field(
        default=[],
        description="Historique de conversation optionnel.",
    )


class SourceDocument(BaseModel):
    source: str = Field(..., description="Nom du texte de loi ou de la convention.")
    article: str = Field(default="", description="Numéro ou titre de l'article.")
    content_preview: str = Field(default="", description="Extrait du texte source.")


class AnswerResponse(BaseModel):
    question: str = Field(..., description="Question posée par l'utilisateur.")
    answer: str = Field(..., description="Réponse générée par le système RAG.")
    sources: List[SourceDocument] = Field(
        default=[],
        description="Documents sources utilisés pour la réponse.",
    )
    intent: str = Field(default="", description="Intent détecté (LEGAL_RAG, GREETING, etc.).")
    latency_ms: float = Field(default=0.0, description="Temps de traitement en millisecondes.")


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "french-labor-law-rag"
    qdrant_connected: bool = False
