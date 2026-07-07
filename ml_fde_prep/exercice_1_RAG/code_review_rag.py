"""
support_bot.py

Assistant de support client : répond aux questions des utilisateurs en s'appuyant
sur la base de connaissance interne (articles d'aide, FAQ, procédures).

Utilisé par l'équipe support pour réduire le nombre de tickets traités manuellement.
"""

import os
import random

import numpy as np


# --- MOCK : simule le SDK Mistral pour que ce fichier tourne sans clé API / dépendance réseau. ---
# En conditions réelles d'entretien, ceci serait `from mistralai import Mistral` avec un vrai client.
class _MockEmbeddingsResp:
    def __init__(self, vec):
        self.data = [type("Emb", (), {"embedding": vec})()]


class _MockChatResp:
    def __init__(self, text):
        self.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": text})()})()
        ]


class Mistral:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = self
        self.chat = self

    def create(self, model, inputs):
        random.seed(hash(inputs[0]) % (2**32))
        return _MockEmbeddingsResp([random.random() for _ in range(16)])

    def complete(self, model, messages):
        question = messages[0]["content"].split("question suivante du client")[-1]
        return _MockChatResp(
            f"[réponse simulée du modèle '{model}'] Voici une réponse basée sur le contexte fourni, en lien avec : {question.strip()[:120]}"
        )


# --- fin du mock ---

client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", "dummy-key-for-practice"))

# Chargement de la base de connaissance (articles d'aide au format texte)
DOCS_FOLDER = "./knowledge_base"


def load_docs():
    docs = []
    for fname in os.listdir(DOCS_FOLDER):
        with open(os.path.join(DOCS_FOLDER, fname)) as f:
            content = f.read()
        # on découpe en chunks de 500 caractères
        for i in range(0, len(content), 500):
            docs.append(content[i : i + 500])
    return docs


def embed(text):
    resp = client.embeddings.create(model="mistral-embed", inputs=[text])
    return resp.data[0].embedding


def answer_question(user_id, question):
    print("Nouvelle question de " + user_id + " : " + question)

    # à chaque question, on recharge et on ré-embed toute la base de connaissance
    docs = load_docs()
    doc_embeddings = []
    for d in docs:
        try:
            doc_embeddings.append(embed(d))
        except:
            pass

    q_embedding = embed(question)

    # calcul de similarité cosinus avec la base
    scores = []
    for i in range(len(doc_embeddings)):
        a = np.array(doc_embeddings[i])
        b = np.array(q_embedding)
        score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        scores.append((score, docs[i]))

    scores.sort(key=lambda x: x[0], reverse=True)
    top_docs = [d for s, d in scores[:5]]

    context = ""
    for d in top_docs:
        context = context + d + "\n---\n"

    prompt = (
        "Tu es un assistant de support client. Voici des extraits de notre documentation:\n\n"
        + context
        + "\n\nRéponds à la question suivante du client de manière claire et concise:\n"
        + question
    )

    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = chat_response.choices[0].message.content

    # log basique pour debug
    print("Réponse générée: " + answer[:100])

    # on enregistre le ticket comme "résolu" automatiquement
    ticket = {
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "status": "resolved",
    }
    save_ticket(ticket)

    return answer


def save_ticket(ticket):
    # TODO: brancher sur le vrai système de ticketing (Zendesk)
    with open("tickets_log.txt", "a") as f:
        f.write(str(ticket) + "\n")


if __name__ == "__main__":
    q = "Comment réinitialiser mon mot de passe ?"
    print(answer_question("user_123", q))
    q = "Comment réinitialiser mon mot de passe ?"
    print(answer_question("user_123", q))
