"""
Module de génération des réponses juridiques.
Gère l'interaction avec le LLM, la construction du prompt et le formatage.
"""

from typing import List, Tuple

from app.core.config import settings
from app.core.logging import get_logger, log_execution_time
from app.rag.prompts import AGGREGATION_PROMPT
from app.schemas.qa import SourceDocument

logger = get_logger(__name__)

_llm = None
_STOP_TOKENS = [chr(60) + "|endoftext|" + chr(62), "Human:", "user:"]


def get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        logger.info(
            f"Initialisation du LLM : {settings.LLM_MODEL} @ {settings.LLM_BASE_URL}"
        )
        _llm = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            streaming=True,
        ).bind(stop=_STOP_TOKENS)
    return _llm


class LegalGenerator:

    def __init__(self):
        self.llm = get_llm()

    def build_context(self, docs: list) -> Tuple[str, List[SourceDocument], str]:
        if not docs:
            return "", [], ""

        grouped_context = {}
        for doc in docs:
            source_name = doc.metadata.get(
                "Livre", doc.metadata.get("source", "Inconnu")
            )
            article_name = doc.metadata.get("Article", "Article Inconnu")

            group_key = f"SOURCE: {source_name}\nARTICLE: {article_name}"
            if group_key not in grouped_context:
                grouped_context[group_key] = []
            grouped_context[group_key].append(doc.page_content.strip())

        context_parts = []
        source_display = "### EXTRAITS JURIDIQUES UTILISÉS :\n\n"
        sources = []

        for key, contents in grouped_context.items():
            stitched_content = "\n[...]\n".join(contents)
            context_parts.append(f"{key}\nTEXTE:\n{stitched_content}")

            parts = key.split("\n")
            src = parts[0].replace("SOURCE: ", "") if len(parts) > 0 else "Inconnu"
            art = parts[1].replace("ARTICLE: ", "") if len(parts) > 1 else ""

            sources.append(
                SourceDocument(
                    source=src,
                    article=art,
                    content_preview=stitched_content[:300],
                )
            )

            ui_title = key.replace("\n", " | ")
            source_display += (
                f"<details><summary style=\"cursor: pointer; font-weight: bold; "
                f"color: #60a5fa;\">{ui_title}</summary>\n\n"
                f"> {stitched_content}\n\n</details>\n<hr>\n"
            )

        final_context = "\n\n---\n\n".join(context_parts)
        return final_context, sources, source_display

    @log_execution_time(logger)
    def generate(self, question: str, docs: list) -> Tuple[str, List[SourceDocument], str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        context, sources, source_display = self.build_context(docs)

        if not context:
            from app.rag.prompts import NO_DOCUMENTS_RESPONSE
            return NO_DOCUMENTS_RESPONSE, [], "Aucun document pertinent trouvé."

        messages = [
            SystemMessage(content=AGGREGATION_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]

        answer = self.llm.invoke(messages).content.strip()
        logger.info(f"Réponse générée ({len(answer)} chars) avec {len(sources)} sources")

        return answer, sources, source_display
