import os
import json
import gradio as gr
import warnings

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

from qdrant_client.http import models as qmodels
import configs
from ingest_db import get_vector_base 

warnings.filterwarnings("ignore")

print("Initialisation de la base de données et des Retrievers...")

# Qdrant Native Hybrid Search
vector_store = get_vector_base()
hybrid_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
cross_encoder = HuggingFaceCrossEncoder(model_name=configs.RERANKER_MODEL_NAME)
compressor = CrossEncoderReranker(model=cross_encoder, top_n=7)

advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=hybrid_retriever
)

# LLM (LM STUDIO)
llm = ChatOpenAI(
    base_url=configs.LMSTUDIO_BASE_URL, 
    api_key="lm-studio",                
    model=configs.LMSTUDIO_MODEL,       
    temperature=0.0,
    max_tokens=1500, 
    streaming=True 
).bind(stop=["<|endoftext|>", "Human:", "user:"])

# PROMPTS 

def get_orchestrator_prompt() -> str:
    return """Tu es le Commandant Suprême d'une IA en Droit du Travail Français.
Analyse la question de l'utilisateur et détermine 1) son intention et 2) le type de document à chercher.

<intents>
[GREETING] : Bonjour, merci, ou bavardage général.
[OFF_TOPIC] : Sujets NON liés au droit du travail français, RH ou administration.
[CLARIFICATION] : Question juridique manquant de contexte (Convention, métier manquant). À prioriser !
[LEGAL_RAG] : Questions spécifiques sur le Code du Travail ou les Conventions.
</intents>

<categories_documents>
- "codes" : Questions générales sur le droit (Code du travail, etc.).
- "conventions" : Si l'utilisateur mentionne un secteur, un métier ou un IDCC (ex: Syntec, HCR, Garage).
- "all" : Si la question est floue ou nécessite de chercher partout.
</categories_documents>

Tu DOIS répondre UNIQUEMENT avec un objet JSON valide, sans code markdown autour :
{
    "intent": "[LEGAL_RAG]",
    "target_doc_type": "codes"
}"""

def get_conversation_summary_prompt() -> str:
    return """Tu es un expert en synthèse juridique.
Ta mission : Créer un résumé ultra-concis (1-2 phrases) de la conversation.

<regles>
1. Concentre-toi UNIQUEMENT sur les faits juridiques : statut du salarié, nom de la convention (ex: Syntec, HCR), et le sujet (ex: démission).
2. Ignore les salutations ou le bavardage.
3. Rédige en FRANÇAIS.
4. Si aucun fait juridique n'est présent, retourne une chaîne vide "".
</regles>"""

def get_rewrite_query_prompt() -> str:
    return """Tu es un Expert en Moteurs de Recherche Juridique Français.
Ta mission : Générer EXACTEMENT 4 requêtes de recherche distinctes pour interroger une base vectorielle (Qdrant).

<strategies>
1. Termes juridiques : Convertis le langage familier en jargon juridique (ex: "viré" -> "licenciement").
2. Spécificité : Combine l'action avec le statut (ex: "préavis démission cadre").
3. Convention (IDCC) : Inclus TOUJOURS la convention si elle est mentionnée (ex: "Syntec", "HCR").
4. Structure : Vise les titres de chapitres (ex: "Rupture du contrat").
</strategies>

<regles>
- Génère EXACTEMENT 4 lignes.
- Chaque ligne DOIT commencer par un tiret "- ".
- AUCUNE introduction, AUCUNE conclusion.
- N'ajoute JAMAIS une convention collective (comme HCR ou Syntec) si l'utilisateur ne l'a pas explicitement mentionnée.
- Conserve les mots-clés juridiques importants de la question originale (ex: ne change pas 'télétravail' par d'autres synonymes)
- CRITIQUE : Si la "Query" actuelle aborde un sujet TOTALEMENT NOUVEAU par rapport à la "Conversation", IGNORE la "Conversation". Ne mélange pas d'anciens sujets avec le nouveau !
</regles>

<exemple>
Input: "Je suis dev chez Syntec, c'est quoi mon préavis si je pars ?"
Output:
- Préavis de démission pour un ingénieur cadre
- Convention collective Syntec préavis de départ
- Rupture du contrat de travail durée du préavis
- Conditions de démission et délai-congé Syntec
</exemple>"""

def get_aggregation_prompt() -> str:
    return """Tu es un Avocat Français en Droit du Travail très strict. 
Tu dois répondre à la question de l'utilisateur en utilisant UNIQUEMENT les informations fournies dans la balise <context>.

<regles_critiques>
1. STRICTE BASE DOCUMENTAIRE : Si la réponse ou les règles permettant de la déduire de façon certaine ne sont pas dans le <context>, dis EXACTEMENT : "Désolé, les documents actuels ne contiennent pas cette information."
2. ZÉRO HALLUCINATION : N'invente jamais de durées, de montants ou de numéros d'articles.
3. CONTRÔLE DE L'INDUSTRIE (FATAL) : Si l'utilisateur pose une question sur une convention spécifique (ex: 'HCR'), IGNORE totalement les textes d'autres conventions (ex: 'Automobile'). Fie-toi au nom de la SOURCE.
4. LANGUE : Réponds dans la MÊME LANGUE que la question de l'utilisateur.
5. Vérifie la catégorie professionnelle (ex: Cadre = Ingénieurs et Cadres, ETAM = Employés/Techniciens/Agents de Maîtrise). Applique les règles correspondantes. LIS ATTENTIVEMENT CHAQUE LIGNE des articles fournis avant de tirer une conclusion.
</regles_critiques>

<format_attendu>
- Analyse : (Examine d'abord les textes un par un, cite les morceaux pertinents, repère les exceptions, et explique ta logique étape par étape)
- Réponse Directe : (Oui, Non, Cela dépend, ou Désolé)
- Base Légale : (Cite la SOURCE et l'ARTICLE précis)
</format_attendu>

<context>
{context}
</context>

Rédige ton analyse avant de formuler ta réponse directe."""

# NODES LANGGRAPH

class AgentState(TypedDict): 
    chat_history: List[Dict[str, str]]
    current_query: str
    intent : str
    target_doc_type: str
    conversation_summary: str
    rewritten_queries: List[str]
    raw_docs: List[Any]
    final_answer: str
    source_display: str

def orchestrator_node(state: AgentState) -> dict:
    messages = [
        SystemMessage(content=get_orchestrator_prompt()),
        HumanMessage(content=state["current_query"])
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
    except Exception as e:
        intent = "[LEGAL_RAG]"
        target_doc_type = "all"

    updates = {"intent": intent, "target_doc_type": target_doc_type}

    intent_upper = intent.upper()
    if "GREETING" in intent_upper:
        updates["final_answer"] = "Bonjour ! Comment puis-je vous aider avec vos questions juridiques en droit du travail français ?"
        updates["source_display"] = "Mode conversationnel activé."
    elif "OFF_TOPIC" in intent_upper:
        updates["final_answer"] = "Désolé, je ne peux vous assister que sur des questions liées au droit du travail et aux conventions collectives (ex: contrat, démission, licenciement)."
        updates["source_display"] = "Mode hors-sujet."
    elif "CLARIFICATION" in intent_upper:
        updates["final_answer"] = "Pour vous répondre avec précision, pourriez-vous m'indiquer votre Convention Collective (ex: Syntec, HCR, Automobile) ainsi que votre statut (Cadre, Employé, etc.) ?"
        updates["source_display"] = "Demande de précision."
    
    return updates

def summary_node(state: AgentState):
    chat_history = state.get("chat_history", [])
    if len(chat_history) < 4:
        return {"conversation_summary": ""}

    history_text = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in chat_history[-5:-1]])
    
    messages = [
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=history_text)
    ]
    return {"conversation_summary": llm.invoke(messages).content.strip()}

def rewrite_node(state: AgentState):
    query = state["current_query"]
    summary = state.get("conversation_summary", "")
    
    messages = [
        SystemMessage(content=get_rewrite_query_prompt()),
        HumanMessage(content=f"Conversation: {summary}\nQuery: {query}")
    ]
    raw_lines = llm.invoke(messages).content.split('\n')
    queries = [line[1:].strip() for line in raw_lines if line.strip().startswith("-")]

    if query not in queries:
        queries.insert(0, query)

    return {"rewritten_queries": queries}

def retrieve_node(state: AgentState):
    queries = state.get("rewritten_queries", [])
    target_category = state.get("target_doc_type", "all")
    print(f"Cible de recherche (Filtre) : {target_category}")

    qdrant_filter = None
    if target_category != "all":
        qdrant_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.doc_type", 
                    match=qmodels.MatchValue(value=target_category)
                )
            ]
        )

    all_docs = []
    for q in queries:
        initial_docs = vector_store.similarity_search(query=q, k=60, filter=qdrant_filter)
        if initial_docs:
            reranked_docs = compressor.compress_documents(initial_docs, q)
            all_docs.extend(reranked_docs)

    unique_parents = {}
    for doc in all_docs:
        parent_content = doc.metadata.get("formatted_parent_content", doc.page_content)
        if parent_content not in unique_parents:
            doc.page_content = parent_content
            unique_parents[parent_content] = doc
            
    top_3_chunks = list(unique_parents.values())[:3]
    
    return {"raw_docs": top_3_chunks}

def aggregate_node(state: AgentState):
    docs = state.get("raw_docs", [])
    query = state["current_query"]

    if not docs:
        return {
            "final_answer": "Désolé, aucune information trouvée dans la base de données.",
            "source_display": "Aucun document pertinent trouvé."
        }
    
    grouped_context = {}
    for doc in docs:
        source_name = doc.metadata.get('Livre', doc.metadata.get('source', 'Inconnu'))
        article_name = doc.metadata.get('Article', 'Article Inconnu')
        
        group_key = f"SOURCE: {source_name}\nARTICLE: {article_name}"
        if group_key not in grouped_context:
            grouped_context[group_key] = []
        grouped_context[group_key].append(doc.page_content.strip())

    context_parts = []
    source_display = "### EXTRAITS JURIDIQUES UTILISÉS :\n\n"
    
    for key, contents in grouped_context.items():
        stitched_content = "\n[...]\n".join(contents) 
        context_parts.append(f"{key}\nTEXTE:\n{stitched_content}")
        
        ui_title = key.replace('\n', ' | ')
        source_display += f"<details><summary style='cursor: pointer; font-weight: bold; color: #60a5fa;'>{ui_title}</summary>\n\n> {stitched_content}\n\n</details>\n<hr>\n"

    final_context = "\n\n---\n\n".join(context_parts)

    messages = [
        SystemMessage(content=get_aggregation_prompt()),
        HumanMessage(content=f"Context:\n{final_context}\n\nQuestion: {query}")
    ]
    
    return {
        "final_answer": llm.invoke(messages).content.strip(),
        "source_display": source_display
    }


# LANGGRAPH
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

workflow.add_conditional_edges("orchestrator", route_intent, {"summarize": "summarize", END: END})
workflow.add_edge("summarize", "rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "aggregate")
workflow.add_edge("aggregate", END)

app_graph = workflow.compile()

# UI & STREAMING GRADIO
def add_text(user_message, chat_history):
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": "Analyse de la demande..."})
    return "", chat_history, str(user_message)

def process_chat(chat_history, current_query):
    initial_state = {
        "chat_history": chat_history,
        "current_query": str(current_query),
        "intent": "", "conversation_summary": "", "rewritten_queries": [],
        "raw_docs": [], "final_answer": "", "source_display": "En attente d'extraction..."
    }

    current_source = "En attente..."

    for event in app_graph.stream(initial_state):
        for node_name, state_update in event.items():
            if node_name == "orchestrator":
                if state_update.get("final_answer"): 
                    chat_history[-1]["content"] = state_update["final_answer"]
                    current_source = state_update.get("source_display", "")
                    yield chat_history, current_source
                    return 
                else:
                    intent_val = state_update.get('intent', '[LEGAL]')
                    filter_val = state_update.get('target_doc_type', 'all')
                    chat_history[-1]["content"] = f"Classification : {intent_val}. Filtre ciblé : {filter_val}. Lecture de l'historique..."
                    yield chat_history, current_source

            elif node_name == "summarize":
                chat_history[-1]["content"] = "Génération des requêtes de recherche..."
                yield chat_history, current_source

            elif node_name == "rewrite":
                queries = state_update.get("rewritten_queries", [])
                q_list = "\n".join([f"- {q}" for q in queries])
                chat_history[-1]["content"] = f" Recherche dans la base Qdrant avec :\n{q_list}"
                yield chat_history, current_source

            elif node_name == "retrieve":
                chat_history[-1]["content"] = "Documents trouvés. Assemblage du contexte juridique..."
                yield chat_history, current_source

            elif node_name == "aggregate":
                chat_history[-1]["content"] = state_update.get("final_answer", "Erreur de synthèse.")
                current_source = state_update.get("source_display", "")
                yield chat_history, current_source

# CSS
custom_css = """
.scrollable-box {
    max-height: 550px;
    overflow-y: auto;
    padding: 15px;
    border: 1px solid #374151;
    border-radius: 8px;
    background-color: #1f2937;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Assistant Juridique IA", css=custom_css) as demo:
    gr.Markdown("<h2 style='text-align: center;'>Assistant Juridique IA (Droit du Travail)</h2>")
    
    current_query = gr.State("")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=550, label="Conversation")
            with gr.Row():
                msg = gr.Textbox(placeholder="Posez votre question", show_label=False, scale=4)
                submit_btn = gr.Button("Envoyer", variant="primary", scale=1)
            clear_btn = gr.ClearButton([msg, chatbot], value="Effacer l'historique")
            
        with gr.Column(scale=1):
            source_box = gr.Markdown("Les sources juridiques s'afficheront ici...", elem_classes="scrollable-box")

    msg.submit(add_text, inputs=[msg, chatbot], outputs=[msg, chatbot, current_query], queue=False).then(
        process_chat, inputs=[chatbot, current_query], outputs=[chatbot, source_box]
    )
    
    submit_btn.click(add_text, inputs=[msg, chatbot], outputs=[msg, chatbot, current_query], queue=False).then(
        process_chat, inputs=[chatbot, current_query], outputs=[chatbot, source_box]
    )
    
if __name__ == "__main__":
    demo.launch(share=False)