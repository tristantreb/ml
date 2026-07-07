# Correction — Exercice 1 : RAG (support client)

**⚠️ N'ouvre ce fichier qu'après avoir fait ta review de `code_review_rag.py` seul, dans les conditions du réel (chrono 20-25 min, à l'oral si possible).**

## Contexte produit à clarifier en premier (le point qui t'a été reproché)

Avant de commenter le code, un bon candidat pose le contexte : "Ce bot répond automatiquement aux clients et marque le ticket comme résolu sans validation humaine — quel est le coût d'une mauvaise réponse ici ? Support bas risque (FAQ générale) ou support à enjeu (facturation, résiliation, données perso) ?" Ça change complètement la criticité de plusieurs problèmes ci-dessous (notamment l'auto-résolution sans garde-fou).

## Bugs / correctness

- `except: pass` dans la boucle d'embedding (silencieux) : si l'embedding d'un doc échoue, le doc disparaît silencieusement de la base de connaissance sans aucune trace — un document important peut ne plus jamais remonter, sans que personne ne s'en aperçoive.
- Aucune gestion du cas où `top_docs` est vide ou peu pertinent (score de similarité très faible) : le prompt est quand même envoyé au LLM, qui va probablement halluciner une réponse plausible plutôt que dire "je ne sais pas". Pour un bot de support client, c'est le bug le plus dangereux du fichier.
- Le ticket est marqué `"status": "resolved"` de façon inconditionnelle, même si la réponse est mauvaise ou si aucun document pertinent n'a été trouvé.

## Typing

- Aucune annotation de type nulle part (`def answer_question(user_id, question):`).
- Le "ticket" est un dict brut avec des clés en string — aucune garantie de structure, aucune validation. Un `TypedDict` ou modèle Pydantic (`Ticket(user_id: str, question: str, answer: str, status: Literal["resolved","escalated"], sources: list[str])`) sécuriserait le contrat et forcerait à décider explicitement des statuts possibles.
- `embed()` retourne `Any` implicitement (liste de floats non typée) — pas de `list[float]` ni de dimension attendue documentée.

## Separation of concerns

- Une seule fonction `answer_question` fait tout : chargement des documents, embedding, recherche de similarité, construction du prompt, appel LLM, logging, et écriture du ticket. Impossible de tester une étape indépendamment des autres.
- Découpage attendu : `KnowledgeBaseLoader` (charge + découpe), `Retriever` (embed + recherche, avec un store réutilisable), `PromptBuilder` (construit le prompt à partir du contexte), `LLMClient` (wrapper autour de l'appel du LLM provider), et un orchestrateur fin qui les assemble. Le ticketing (métier support) ne devrait même pas être dans le même module que la logique RAG.

## Maintenabilité

- Concaténation de strings avec `+` pour construire les prompts (`print("Nouvelle question de " + user_id + ...)`) : illisible, fragile, à remplacer par des f-strings, et idéalement par un template de prompt versionné séparément du code (pour pouvoir itérer sur le prompt sans redéployer).
- Magic numbers partout et non documentés : taille de chunk (`500`), `top_k` (`5`), longueur de log tronquée (`100`). Devraient être des constantes nommées ou de la config (fichier/env), avec une justification (pourquoi 500 caractères et pas 300 ou 1000 ?).
- `print()` utilisé comme logging — pas de niveau (info/warning/error), pas de format structuré, impossible à brancher sur un système d'observabilité en prod.
- Nom de modèle `"mistral-large-latest"` en dur — pas de config centralisée, donc changer de modèle nécessite de modifier le code source.
- Le TODO en dur ("brancher sur Zendesk") dans une fonction qui écrit dans un fichier texte local : à la fois un signal que c'est un prototype pas prêt pour la prod, et un risque que ça parte en prod tel quel si personne ne suit ce TODO.

## Scalabilité (point le plus lourd de ce fichier)

- **Le pire problème du fichier** : `load_docs()` et le ré-embedding de **toute** la base de connaissance sont refaits à **chaque question posée**. Avec une base de 10 documents ça passe inaperçu ; avec une base de 10 000 documents en prod, chaque question coûte des milliers d'appels d'embedding avant même de répondre — latence et coût qui explosent linéairement avec la taille de la base, alors que la base change rarement. Il faut un index vectoriel construit une fois (à l'ingestion, en batch ou en incrémental) et réutilisé à chaque requête (FAISS, pgvector, Qdrant, etc.).
- Tout est synchrone et bloquant : embeddings calculés un par un dans une boucle Python, aucun appel en batch alors que l'API d'embedding le permet généralement, aucun `async`/parallélisation. Sous charge (plusieurs utilisateurs simultanés), le service devient un goulot d'étranglement.
- Aucun cache pour les questions fréquentes (beaucoup de questions de support se répètent) ni pour les embeddings de documents inchangés.
- Aucun retry/backoff sur les appels réseau (embedding, chat) : une erreur transitoire de l'API fait planter toute la requête utilisateur au lieu d'être absorbée.
- Écriture des tickets dans un fichier texte local (`tickets_log.txt`) : ne scale pas au-delà d'une instance unique, pas de concurrence gérée (deux requêtes simultanées peuvent corrompre le fichier).

## Angle produit / business (le point à mettre en avant à l'oral)

- Un bot qui **auto-résout** un ticket de support sans donner les sources utilisées casse la confiance et la traçabilité : côté client entreprise, on veut pouvoir auditer "pourquoi le bot a répondu ça" — il faut retourner les documents sources avec la réponse (citations), pas juste le texte généré.
- Aucune métrique business définie : taux de résolution réelle (le client a-t-il rouvert un ticket derrière ?), taux d'escalade vers un humain, satisfaction. Sans ces métriques, impossible de démontrer l'impact au client — exactement le type de lien technique → business à faire spontanément à l'oral.
- Pas de mécanisme de fallback vers un humain quand la confiance est faible : côté business, mieux vaut escalader un cas incertain que de risquer une mauvaise réponse marquée "résolue" qui dégrade la satisfaction client et la confiance dans le produit.

## Comment prioriser à l'oral

1. **Bloquant avant prod** : absence de fallback/seuil de confiance + auto-résolution inconditionnelle (risque business direct), ré-embedding de toute la base à chaque requête (ne tient pas la charge).
2. **Important, à planifier** : séparation en modules, typage, gestion d'erreurs explicite, logging structuré, cache/retry.
3. **Nice-to-have** : f-strings à la place des concaténations, extraction des magic numbers en config.

# Point ajoutés par moi
- Use conversations API (beta) to get message history on server side with `conversation_id`
- Limitation de context window
- Current `mistral-large-latest` released in De 2025, 256k context window, $0.50/M inputs, $1.5/M output (note GPT5.5 $5/$30)