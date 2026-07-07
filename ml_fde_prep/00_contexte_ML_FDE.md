# Contexte : Forward Deployed ML Engineer

## Le rôle en une phrase

Le FDE (Forward Deployed Engineer), c'est l'ingénieur qui va chez le client, comprend son problème business, et livre une solution IA en prod — du premier call de pré-vente jusqu'à l'industrialisation. La JD le dit explicitement : "operating like startup CTOs who own end-to-end project execution". Ce n'est pas un rôle de recherche, ni un rôle purement produit interne : c'est un rôle hybride tech + business + relationnel.

Trois choses sont évaluées en continu chez un FDE :
1. La compétence technique (ML/LLM, code de prod, APIs)
2. La compréhension business (quel problème on résout, quel impact ça a, pour qui)
3. La capacité à communiquer avec des interlocuteurs non-techniques

Ton feedback de screening ("trop de solutions techniques, pas assez de problèmes/impact") pointe directement le point 2. C'est le fil rouge à travailler pour tout le reste du process, y compris à l'oral pendant le code review (voir plus bas — même un exercice de code se juge aussi sur ce point).

## Le cycle de vie type d'un projet FDE

Un projet FDE typique traverse ces phases. Utile de les avoir en tête pour recontextualiser n'importe quel bout de code qu'on te montre — un reviewer sérieux se demande toujours "à quelle étape ce fichier appartient, et qu'est-ce que ça implique comme contraintes ?"

**1. Discovery / avant-vente**
Comprendre le vrai problème du client : quel process manuel/coûteux/lent veut-on remplacer, quel est le volume (10 requêtes/jour ou 100k ?), quelles contraintes (données sensibles, latence, budget, compliance sectorielle). Sortir de cette phase avec des métriques cibles concrètes (temps de traitement, taux de résolution, coût par requête), pas juste "on veut de l'IA".

**2. Design de la solution**
Choix d'architecture en fonction du problème, pas l'inverse : RAG (besoin de connaissance propriétaire non gelée dans le modèle), fine-tuning (besoin de style/format très spécifique, volume de données suffisant), agentique (besoin d'orchestrer plusieurs systèmes/actions), ou simplement du prompt engineering bien fait. Un FDE senior sait dire "on n'a pas besoin d'un agent ici, un simple call structuré suffit" — la sobriété d'architecture est un signal de maturité.

**3. Prototype (PoC)**
Script ou notebook qui prouve la faisabilité sur des données réelles du client, en quelques jours. Le code peut être sale à ce stade — c'est normal et attendu, ce n'est pas encore le sujet de l'entretien.

**4. Industrialisation (= le sujet de l'entretien code review)**
Le PoC devient un service : API backend (FastAPI/Flask en général), séparation claire des couches (ingestion des données / retrieval / construction du prompt / appel LLM / logique métier / formatage de la réponse), gestion d'erreurs explicite, logging structuré, configuration externalisée (pas de magic numbers ni de clés en dur), typage strict (Pydantic/type hints) pour que l'API ait un contrat clair avec le frontend et les autres services.

**5. Intégration frontend / consommation**
Le résultat est consommé par une UI (chat interne, plugin, workflow), donc le contrat d'API doit être stable et versionné. C'est souvent une équipe produit du client qui branche ça — donc la doc et le typage ne sont pas du confort, ils conditionnent si l'intégration se fait en 1 jour ou en 2 semaines.

**6. Scalabilité / mise en production réelle**
Passage de la démo (1 utilisateur, requêtes séquentielles) à la charge réelle : async, gestion de la concurrence, caching (embeddings, réponses fréquentes), retries/backoff sur les appels API externes, rate limiting, coût token maîtrisé, observabilité (traces, latence, taux d'erreur par étape).

**7. Mesure d'impact & itération**
Un FDE doit pouvoir répondre à "est-ce que ça marche mieux qu'avant, et de combien ?" avec des chiffres business (heures économisées, taux de résolution, satisfaction), pas juste "le modèle répond bien". Boucle de feedback avec le client pour itérer.

**8. Hand-off / maintenance**
Documentation, tests, transfert vers les équipes internes du client ou le support. Le code doit survivre au départ de son auteur.

## Pourquoi le code review porte sur maintenabilité + scalabilité en particulier

Un FDE écrit du code qui sera repris par les équipes du client, tourne en prod chez une entreprise (pas dans un labo), et doit encaisser une montée en charge non prévisible au moment du design initial. D'où l'accent mis sur :

- **Typing** : contrat clair, erreurs détectées à la lecture/à la compilation plutôt qu'en prod chez le client.
- **Separation of concerns** : un fichier avec "un appel LLM et un peu de logique autour" révèle immédiatement si le candidat sait isoler la couche IO (API externe), la couche métier (règles, décisions), et la couche présentation (formatage de sortie).
- **Maintenabilité** : est-ce que ce code peut être repris par un autre ingénieur (ou toi dans 6 mois) sans tout redécouvrir ? Nommage, config externalisée, pas de duplication, gestion d'erreurs explicite plutôt que des `except: pass` silencieux.
- **Scalabilité** : est-ce que ce qui marche pour 1 utilisateur en démo tient pour 1000 utilisateurs en prod chez un client entreprise ? Async, statelessness, pas de ressource recréée à chaque appel, pas de mémoire qui grossit sans borne.

## Comment aborder l'exercice de code review (méthode)

Structure ta review en 4 passes, à l'oral, dans cet ordre — ça montre une progression de raisonnement plutôt qu'une liste de remarques en vrac :

1. **Comprendre le contexte produit avant de plonger dans le code.** Pose la question : "à qui sert ce code, quel est le use case, quel volume/criticité ?" si ce n'est pas donné. Un reviewer qui fonce dans le detail sans clarifier le contexte reproduit exactement le biais pointé dans ton screening.
2. **Correctness / bugs fonctionnels** : est-ce que le code fait ce qu'il prétend faire, y a-t-il des cas non gérés (réponse vide, erreur API, entrée malformée) ?
3. **Design & qualité** : typing, separation of concerns, maintenabilité — avec des propositions concrètes de refactor (pas juste "c'est mal fait").
4. **Production-readiness & impact business** : scalabilité, coûts, observabilité, et surtout — relie chaque remarque technique à une conséquence concrète pour le client ("si ce cache n'existe pas, chaque requête réembed toute la base, donc à 500 req/jour le coût explose et la latence aussi, ce qui dégrade l'expérience utilisateur final").

Termine toujours par une priorisation : qu'est-ce qui bloque la mise en prod (bloquant), qu'est-ce qui est important mais pas bloquant (à planifier), qu'est-ce qui est du nice-to-have. C'est ce réflexe de priorisation, plus que la liste exhaustive de problèmes, qui distingue un FDE d'un simple reviewer de code.

## Les 3 exercices fournis

| Fichier | Use case | Pattern |
|---|---|---|
| `exercice_1_RAG/code_review_rag.py` | Support client (FAQ interne) | RAG classique |
| `exercice_2_agentique/code_review_agent.py` | Agent d'onboarding client orchestrant CRM/billing/email | Agentique multi-API |
| `exercice_3_MCP/code_review_mcp.py` | Copilote support connecté à un outil interne via MCP | Connecteur MCP |

Chaque fichier de code contient des problèmes injectés volontairement sur les 4 axes (typing, SoC, maintenabilité, scalabilité) avec un accent plus fort sur maintenabilité et scalabilité, plus des angles "produit/business" à repérer. Les corrections sont dans des fichiers séparés (`correction_*.md`) pour que tu puisses t'entraîner en conditions réelles d'abord — ouvre-les seulement après ton passage.
