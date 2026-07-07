# Correction — Exercice 3 : Copilote connecté via MCP

**⚠️ N'ouvre ce fichier qu'après avoir fait ta review de `code_review_mcp.py` seul, chrono 20-25 min.**

## Contexte produit à clarifier en premier

C'est un outil interne (ingénieurs, pas clients finaux) mais qui touche à des données de tickets potentiellement sensibles (incidents, clients cités dans les descriptions). Un bon candidat demande : "Est-ce que tous les ingénieurs ont accès à tous les tickets Jira, ou y a-t-il des projets restreints ? Ce copilote respecte-t-il les mêmes permissions que l'utilisateur qui pose la question dans Slack, ou interroge-t-il Jira avec un compte de service qui voit tout ?" C'est la question de sécurité/business la plus importante du fichier, et elle n'a pas de réponse dans le code actuel.

## Bugs / correctness

- `[w for w in question.split() if w.startswith("INFRA-")][0]` plante avec un `IndexError` si la question contient "INFRA-" sans qu'un mot commence exactement par ce préfixe après split, ou si l'utilisateur écrit "le ticket infra-482" en minuscules — extraction de clé de ticket beaucoup trop fragile pour une vraie logique métier (à remplacer par un vrai parsing / regex, ou mieux, laisser le LLM extraire la clé via function calling structuré plutôt qu'un parsing de string côté code).
- Aucune gestion du cas où `mcp.call_tool` lève une exception (ticket inexistant, erreur serveur MCP) — le programme plante avec une stack trace brute au lieu d'un message utile envoyé dans Slack.

## Typing

- Le résultat de `mcp.call_tool` est un dict brut accédé par clé (`result["key"]`, `result["assignee"]`...) sans validation — si le serveur MCP change son schéma de réponse ou renvoie un champ manquant, `KeyError` en prod sans message clair. Un modèle Pydantic (`JiraIssue(key: str, status: str, assignee: str, summary: str)`) validerait la forme dès la sortie de l'appel MCP.
- Pas de type de retour documenté pour `handle_slack_question` (retourne un dict `slack_message` non typé).
- `tools` (retour de `list_tools()`) est une liste de dicts non typée, simplement injectée telle quelle dans le prompt système via f-string (`f"...{tools}"`) — fragile et impossible à valider.

## Separation of concerns

- `handle_slack_question` mélange quatre responsabilités : gestion de la connexion MCP, construction du prompt, appel LLM + dispatch de l'outil, et formatage du message Slack. Découpage attendu : un client MCP encapsulé à part (connexion + cache des tools), un module de construction de prompt, un orchestrateur LLM+tool-calling générique, et un formateur de sortie Slack séparé (le format de sortie changerait si demain le même copilote devait aussi répondre par email ou API REST).
- L'état global `session` (variable module-level mutée par `get_mcp_session`) est un couplage fort entre la connexion MCP et le reste du code — aucune fonction n'est testable indépendamment sans passer par cet état partagé.
- Le formatage du texte de réponse (`"Ticket " + result["key"] + " : " + ...`) est câblé en dur dans la fonction d'orchestration alors que c'est une responsabilité de présentation, indépendante de la logique d'appel MCP/LLM.

## Maintenabilité

- `os.environ["JIRA_MCP_SERVER_URL"]` et `os.environ["JIRA_MCP_TOKEN"]` sont lus directement sans valeur par défaut ni message d'erreur explicite — si une variable manque en prod, le candidat aura un `KeyError: 'JIRA_MCP_TOKEN'` brut, sans indication claire de configuration manquante pour la personne qui déploie.
- Les noms d'outils MCP (`"get_issue"`, `"search_issues"`) sont des strings en dur répétés entre le mock du serveur et le dispatch — si le serveur MCP renomme un outil, ou si on branche un second serveur MCP (Confluence, GitHub...) demain, tout le code est à réécrire car il est fortement couplé aux noms et à la forme de réponse de ce serveur Jira spécifique en particulier.
- Que des `print()` pour tracer les appels MCP — pas de logging structuré, impossible de distinguer en prod un appel qui a réussi lentement d'un appel qui a échoué.
- Aucun test, aucune doc sur le format attendu des réponses des outils MCP.

## Scalabilité

- **Le problème le plus lourd du fichier** : `get_mcp_session()` recrée une connexion MCP à **chaque question posée** dans Slack (`session = MockMCPSession(...)` à chaque appel de `handle_slack_question`). En prod, avec plusieurs questions par minute de la part de l'équipe ingénieurs, ça multiplie inutilement le coût/latence d'établissement de connexion — il faut une connexion (ou un pool) réutilisée entre les requêtes, initialisée une fois au démarrage du service.
- `mcp.list_tools()` est rappelé à chaque requête alors que la liste d'outils exposés par un serveur MCP change rarement — devrait être mis en cache (avec invalidation périodique ou au redémarrage), pas recalculé à chaque question.
- Tout est synchrone/bloquant : aucun `async`, alors qu'un client MCP fait fondamentalement des appels réseau — sous plusieurs requêtes Slack simultanées (plusieurs ingénieurs posent une question en même temps), le service traite tout en séquentiel.
- La variable globale `session` partagée entre requêtes sans aucune protection de concurrence est un risque de race condition si le service traite des requêtes en parallèle (deux threads/coroutines pourraient se marcher dessus sur le même objet `session`).
- Pas de gestion du cas où le serveur MCP est indisponible : aucun fallback, aucun circuit breaker — une panne du serveur MCP Jira devient un point de défaillance unique qui casse tout le copilote Slack, alors que ce n'est qu'une des sources de données possibles.

## Angle produit / business (le point à mettre en avant à l'oral)

- **Pas de scoping par utilisateur/permissions** : le copilote interroge Jira avec un unique token de service, sans propager l'identité de la personne qui pose la question dans Slack — risque business et sécurité réel si certains projets Jira sont confidentiels (RH, sécurité, clients spécifiques) et que le copilote peut quand même en extraire des infos à n'importe quel ingénieur.
- **Pas de traçabilité de la source** : la réponse Slack ne précise pas explicitement "info récupérée via Jira, ticket INFRA-482, à l'instant T" de façon structurée/horodatée — pour un outil interne ça semble mineur, mais si demain ce pattern MCP est réutilisé pour un copilote client-facing, l'absence d'audit trail devient bloquante (cf. exercice agentique).
- **Aucune gestion d'échec orientée utilisateur** : si le serveur MCP tombe ou que le ticket n'existe pas, l'ingénieur dans Slack ne reçoit qu'un plantage ou un message générique, sans indication actionnable ("le serveur Jira MCP est indisponible, réessayez dans quelques minutes" vs. une stack trace) — dégrade la confiance dans l'outil et génère plus de charge support que le manuel.
- Pas de mesure de valeur (nombre de questions traitées automatiquement vs. nécessitant d'aller chercher dans Jira manuellement) — pourtant c'est exactement le chiffre qui justifierait d'investir dans ce type d'outil interne auprès d'un manager.

## Comment prioriser à l'oral

1. **Bloquant avant prod** : reconnexion MCP à chaque requête (coût/latence), absence de scoping des permissions par utilisateur, absence de gestion d'erreur MCP (single point of failure).
2. **Important, à planifier** : extraction typée du résultat MCP (Pydantic), découplage du code vis-à-vis du schéma spécifique de ce serveur MCP, cache de `list_tools()`.
3. **Nice-to-have** : passage à l'async, logging structuré, meilleur parsing de la clé de ticket.
