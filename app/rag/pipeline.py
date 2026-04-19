"""
Pipeline d'orchestration LangGraph pour le système RAG juridique.
Point d'entrée principal : ask_question()
"""

import json
import time
from typing import TypedDict, List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.core.logging import get_logger
from app.rag.prompts import (
    ORCHESTRATOR_PROMPT,
    CONVERSATION_SUMMARY_PROMPT,
    REWRITE_QUERY_PROMPT,
    GREETING_RESPONSE,
    OFF_TOPIC_RESPONSE,
    CLARIFICATION_RESPONSE,
)
from app.rag.generator import get_llm
from app.rag.retriever import LegalRetriever
from app.rag.generator import LegalGenerator
from app.schemas.qa import AnswerResponse, SourceDocument

logger = get_logger(__name__)


class AgentState(TypedDict):
    chat_history: List[Dict[str, str]]
    current_query: str
    intent: str
    target_doc_type: str
    conversation_summary: str
    rewritten_queries: List[str]
    raw_docs: List[Any]
    final_answer: str
    sources: List[dict]
    source_display: str


def orchestrator_node(state: AgentState) -> dict:
    llm = get_llm()
    messages = [
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=state["current_query"]),
    ]
    response_text = llm.invoke(messages).content.strip()

    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text

        parsed_data = json.loads(json_str)
        intent = parsed_data.get("intent", "[LEGAL_RAG]")
        target_doc_type = parsed_data.get("target_doc_type", "all")
    except Exception:
        logger.warning(f"Échec du parsing orchestrateur : {response_text[:100]}")
        intent = "[LEGAL_RAG]"
        target_doc_type = "all"

    logger.info(f"Intent : {intent} | Filtre : {target_doc_type}")
    updates = {"intent": intent, "target_doc_type": target_doc_type}

    intent_upper = intent.upper()
    if "GREETING" in intent_upper:
        updates["final_answer"] = GREETING_RESPONSE
        updates["source_display"] = "Mode conversationnel."
        updates["sources"] = []
    elif "OFF_TOPIC" in intent_upper:
        updates["final_answer"] = OFF_TOPIC_RESPONSE
        updates["source_display"] = "Hors-sujet."
        updates["sources"] = []
    elif "CLARIFICATION" in intent_upper:
        updates["final_answer"] = CLARIFICATION_RESPONSE
        updates["source_display"] = "Demande de clarification."
        updates["sources"] = []

    return updates


def summary_node(state: AgentState) -> dict:
    chat_history = state.get("chat_history", [])
    if len(chat_history) < 4:
        return {"conversation_summary": ""}

    llm = get_llm()
    history_text = "\n".join(
        [
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history[-5:-1]
        ]
    )
    messages = [
        SystemMessage(content=CONVERSATION_SUMMARY_PROMPT),
        HumanMessage(content=history_text),
    ]
    summary = llm.invoke(messages).content.strip()
    logger.info(f"Résumé de la conversation : {summary[:80]}...")
    return {"conversation_summary": summary}


def rewrite_node(state: AgentState) -> dict:
    llm = get_llm()
    query = state["current_query"]
    summary = state.get("conversation_summary", "")

    messages = [
        SystemMessage(content=REWRITE_QUERY_PROMPT),
        HumanMessage(content=f"Conversation: {summary}\nQuery: {query}"),
    ]
    raw_lines = llm.invoke(messages).content.split("\n")
    queries = [line[1:].strip() for line in raw_lines if line.strip().startswith("-")]

    if query not in queries:
        queries.insert(0, query)

    logger.info(f"Requêtes reformulées : {queries}")
    return {"rewritten_queries": queries}


def retrieve_node(state: AgentState) -> dict:
    queries = state.get("rewritten_queries", [])
    target_category = state.get("target_doc_type", "all")

    retriever = LegalRetriever()
    docs = retriever.retrieve(queries, doc_type_filter=target_category)

    return {"raw_docs": docs}


def aggregate_node(state: AgentState) -> dict:
    docs = state.get("raw_docs", [])
    query = state["current_query"]

    generator = LegalGenerator()
    answer, sources, source_display = generator.generate(query, docs)

    return {
        "final_answer": answer,
        "sources": [s.model_dump() for s in sources],
        "source_display": source_display,
    }


def _build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("summarize", summary_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("aggregate", aggregate_node)

    workflow.set_entry_point("orchestrator")

    def route_intent(state: AgentState):
        if "LEGAL_RAG" in state.get("intent", "").upper():
            return "summarize"
        return END

    workflow.add_conditional_edges(
        "orchestrator", route_intent, {"summarize": "summarize", END: END}
    )
    workflow.add_edge("summarize", "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "aggregate")
    workflow.add_edge("aggregate", END)

    return workflow.compile()


app_graph = _build_graph()


def ask_question(
    question: str,
    chat_history: List[Dict[str, str]] = None,
) -> AnswerResponse:
    start_time = time.perf_counter()
    chat_history = chat_history or []

    logger.info(f"Traitement de la question : {question[:80]}...")

    initial_state = {
        "chat_history": chat_history,
        "current_query": str(question),
        "intent": "",
        "target_doc_type": "all",
        "conversation_summary": "",
        "rewritten_queries": [],
        "raw_docs": [],
        "final_answer": "",
        "sources": [],
        "source_display": "",
    }

    final_state = app_graph.invoke(initial_state)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    response = AnswerResponse(
        question=question,
        answer=final_state.get("final_answer", "Erreur interne."),
        sources=[
            SourceDocument(**s) for s in final_state.get("sources", [])
        ],
        intent=final_state.get("intent", ""),
        latency_ms=round(elapsed_ms, 1),
    )

    logger.info(
        f"Question traitée en {elapsed_ms:.0f}ms | "
        f"Intent : {response.intent} | Sources : {len(response.sources)}"
    )

    return response

