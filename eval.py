import json
import time
from groq import Groq
from app_langgraph import app_graph, orchestrator_node, rewrite_node, retrieve_node, aggregate_node
import configs

client = Groq(api_key=configs.GROQ_API_KEY)
eval_dataset = [
    {
        "test_type": "Direct Extraction",
        "question": "Un accident survenu à mon domicile pendant mes heures de télétravail est-il considéré comme un accident du travail ?",
        "golden_context": "SOURCE: Code du travail\nARTICLE: Article L1222-9\nTEXTE: L'accident survenu sur le lieu où est exercé le télétravail pendant l'exercice de l'activité professionnelle du télétravailleur est présumé être un accident de travail...",
        "ground_truth": "Oui. Selon l'article L1222-9 du Code du travail, un tel accident est présumé être un accident de travail."
    },
    {
        "test_type": "Conditional Logic",
        "question": "Mon employeur peut-il refuser ma demande de télétravail alors que je suis un travailleur handicapé ?",
        "golden_context": "SOURCE: Code du travail\nARTICLE: Article L1222-9\nTEXTE: Lorsque la demande de recours au télétravail est formulée par un travailleur handicapé... l'employeur motive, le cas échéant, sa décision de refus.",
        "ground_truth": "Oui, il peut refuser, mais selon l'article L1222-9, s'agissant d'un travailleur handicapé, l'employeur a l'obligation légale de motiver sa décision de refus."
    },
    {
        "test_type": "Exception Trap",
        "question": "L'entreprise qui m'emploie vient d'être rachetée. Le nouvel employeur peut-il modifier mon ancienneté ou refuser de reprendre mon contrat ?",
        "golden_context": "SOURCE: Code du travail\nARTICLE: Article L1224-1\nTEXTE: Lorsque survient une modification dans la situation juridique de l'employeur, notamment par succession, vente, fusion... tous les contrats de travail en cours au jour de la modification subsistent entre le nouvel employeur et le personnel.",
        "ground_truth": "Non. L'article L1224-1 stipule que tous les contrats en cours subsistent automatiquement avec le nouvel employeur."
    },
    {
        "test_type": "Scenario",
        "question": "Je suis en congé maternité. Mon employeur vient de découvrir que j'ai commis une faute grave avant mon départ. Peut-il me licencier et rompre le contrat immédiatement ?",
        "golden_context": "SOURCE: Code du travail\nARTICLE: Article L1225-4\nTEXTE: Toutefois, l'employeur peut rompre le contrat s'il justifie d'une faute grave... Dans ce cas, la rupture du contrat de travail ne peut prendre effet ou être notifiée pendant les périodes de suspension du contrat de travail...",
        "ground_truth": "Il peut justifier un licenciement pour faute grave non liée à la grossesse. Cependant, selon l'article L1225-4, la rupture du contrat ne peut ni prendre effet ni vous être notifiée pendant votre congé maternité."
    },
    {
        "test_type": "Financial Logic",
        "question": "Mon employeur me demande de suivre une formation à la sécurité incendie un samedi, en dehors de mes heures de travail, et sans me payer. Est-ce légal ?",
        "golden_context": "SOURCE: Code du travail\nARTICLE: Article R4141-5\nTEXTE: Le temps consacré à la formation et à l'information... est considéré comme temps de travail. La formation et l'information en question se déroulent pendant l'horaire normal de travail.",
        "ground_truth": "C'est illégal. Selon l'article R4141-5, cette formation doit se dérouler pendant l'horaire normal de travail et est considérée comme du temps de travail rémunéré."
    },
    {
        "test_type": "Logical Reasoning",
        "question": "J'ai déposé un recours contre une mise en demeure il y a 25 jours et le directeur régional ne m'a pas répondu. Mon recours est-il rejeté ?",
        "golden_context": "SOURCE: Code du travail\nARTICLE: Article R4723-4\nTEXTE: La non-communication à l'employeur de la décision... dans le délai de 21 jours vaut acceptation du recours.",
        "ground_truth": "Non, au contraire. D'après l'article R4723-4, le silence du directeur au-delà du délai de 21 jours vaut acceptation de votre recours."
    },
    {
        "test_type": "Data Extraction",
        "question": "Quels sont les objectifs fixés par la loi concernant l'écart des pensions de retraite entre les hommes et les femmes pour 2037 ?",
        "golden_context": "SOURCE: Code de la sécurité sociale\nARTICLE: Article L111-2-1\nTEXTE: Elle se fixe pour objectifs... à l'horizon 2037, sa réduction de moitié par rapport à l'écart constaté en 2023.",
        "ground_truth": "Selon l'article L111-2-1, l'objectif pour 2037 est la réduction de moitié de l'écart constaté en 2023."
    },
    {
        "test_type": "Exception Trap",
        "question": "L'URSSAF a débarqué dans mon entreprise sans m'avoir envoyé d'avis 30 jours à l'avance. Le contrôle est-il nul ?",
        "golden_context": "SOURCE: Code de la sécurité sociale\nARTICLE: Article R243-59\nTEXTE: Toutefois, l'organisme n'est pas tenu à cet envoi dans le cas où le contrôle est effectué pour rechercher des infractions aux interdictions mentionnées à l'article L. 8221-1 du code du travail (travail dissimulé).",
        "ground_truth": "Pas nécessairement. Si le contrôle a pour but de rechercher des infractions pour travail dissimulé, l'article R243-59 précise que l'envoi de l'avis préalable n'est pas obligatoire."
    },
    {
        "test_type": "Process Timeline",
        "question": "Combien de temps ai-je pour accepter l'offre d'indemnisation du fonds pour mon enfant victime de pesticides, et que se passe-t-il si je l'accepte ?",
        "golden_context": "SOURCE: Code de la sécurité sociale\nARTICLE: Article R491-8\nTEXTE: Dans un délai de quatre mois... L'acceptation de l'offre vaut transaction au sens de l'article 2044 du code civil.",
        "ground_truth": "Vous avez 4 mois pour l'accepter. Selon l'article R491-8, cette acceptation vaut transaction (accord définitif)."
    },
    {
        "test_type": "Mathematical Condition",
        "question": "Nous fabriquons et posons des enseignes lumineuses (100 salariés). 50 font la fabrication et 50 font la pose. Doit-on obligatoirement appliquer la convention du Bâtiment ?",
        "golden_context": "SOURCE: Convention Bâtiment\nARTICLE: Clause d'attribution\nTEXTE: 2. Lorsque le personnel concourant à la pose au sens ci-dessus se situe entre 20 % et 80 %, les entreprises peuvent opter...",
        "ground_truth": "Non. Puisque le personnel de pose représente 50% (entre 20% et 80%), l'entreprise peut opter pour une autre convention après accord avec les représentants du personnel."
    },
    {
        "test_type": "Direct Extraction",
        "question": "Je suis embauché comme Cadre sous la convention Syntec. Quelle est la durée de ma période d'essai ?",
        "golden_context": "SOURCE: Convention Syntec\nARTICLE: Article 3.4\nTEXTE: ...ingénieurs et cadres, la période d'essai est de 4 mois maximum. Elle peut être renouvelée pour une durée de 4 mois maximum.",
        "ground_truth": "La période d'essai est de 4 mois maximum, renouvelable une fois pour 4 mois."
    }
]

def get_groq_evaluation(question, golden_context, ground_truth, predicted_answer):
    prompt = f"""Tu es un juge impartial et un expert en Droit du Travail français.
Ta mission est d'évaluer la réponse fournie par un système d'IA (Candidat) en la comparant à la réponse parfaite (Ground Truth).

[QUESTION]
{question}

[CONTEXTE LÉGAL (GOLDEN CONTEXT)]
{golden_context}

[RÉPONSE PARFAITE (GROUND TRUTH)]
{ground_truth}

[RÉPONSE DU CANDIDAT (PREDICTED ANSWER)]
{predicted_answer}

[CRITÈRES D'ÉVALUATION]
Note la réponse du candidat de 0 à 5 selon ces critères :
- 5/5 : Parfait. Sens identique à la réponse parfaite, cite les bons articles, aucune hallucination.
- 4/5 : Très bon. Sens correct, mais manque un petit détail ou une référence.
- 3/5 : Passable. Réponse partiellement correcte mais omet une condition importante ou manque de clarté.
- 1-2/5 : Faux. Contredit la loi ou répond à côté de la plaque.
- 0/5 : Hallucination grave (invente des lois, des délais ou des articles qui n'existent pas).

Évalue UNIQUEMENT la précision des faits et du droit. Ne pénalise pas si la formulation est différente.
Tu DOIS renvoyer ta réponse au format JSON valide avec exactement deux clés : "score" (un nombre entier de 0 à 5) et "reason" (une explication courte). N'ajoute aucun texte avant ou après le JSON.
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}, 
            temperature=0.0,
        )
        response_text = chat_completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        print(f"Erreur lors de l'appel à Groq : {e}")
        return {"score": 0, "reason": "Erreur API Groq"}


def get_qwen_diagnostics(question):
    print(f"Interrogation de l'IA locale (Qwen) : {question}")
    
    state = {
        "current_query": question, 
        "chat_history": [], 
        "conversation_summary": ""
    }
    state.update(orchestrator_node(state))
    state.update(rewrite_node(state))
    rewritten_queries = state.get("rewritten_queries", [])
    
    state.update(retrieve_node(state))
    raw_docs = state.get("raw_docs", [])
    retrieved_texts = [doc.page_content for doc in raw_docs] if raw_docs else []
    
    state.update(aggregate_node(state))
    final_answer = state.get("final_answer", "Erreur : Aucune réponse trouvée.")
                
    return {
        "final_answer": final_answer,
        "used_queries": rewritten_queries,
        "retrieved_context": retrieved_texts
    }


def run_evaluation():
    print(f"Début de l'évaluation de {len(eval_dataset)} questions...")
    results = []
    total_score = 0
    
    for index, test_case in enumerate(eval_dataset):
        print(f"\n--- Question {index + 1}/{len(eval_dataset)} : [{test_case['test_type']}] ---")
        diagnostics = get_qwen_diagnostics(test_case['question'])
        predicted_answer = diagnostics["final_answer"]
        print(f"Réponse générée : {predicted_answer[:100]}...")
        
        print("Évaluation par Groq en cours...")
        eval_result = get_groq_evaluation(
            test_case['question'],
            test_case['golden_context'],
            test_case['ground_truth'],
            predicted_answer
        )
        
        score = eval_result.get("score", 0)
        reason = eval_result.get("reason", "")
        
        print(f"Score : {score}/5 | Raison : {reason}")
        
        total_score += score
        results.append({
            "question": test_case['question'],
            "score": score,
            "reason": reason,
            "predicted_answer": predicted_answer,
            "ground_truth": test_case['ground_truth'],
            "diagnostic_queries": diagnostics["used_queries"],      
            "diagnostic_context": diagnostics["retrieved_context"]  
        })
        time.sleep(2) 

    max_score = len(eval_dataset) * 5
    accuracy = (total_score / max_score) * 100
    print(f"\n================================================")
    print(f"ÉVALUATION TERMINÉE !")
    print(f"Score total : {total_score}/{max_score} ({accuracy:.2f}%)")
    
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Les résultats détaillés ont été enregistrés dans le fichier 'evaluation_results.json'.")

if __name__ == "__main__":
    run_evaluation()