# Correction — Exercice 2 : Agent d'onboarding multi-API

**⚠️ N'ouvre ce fichier qu'après avoir fait ta review de `code_review_agent.py` seul, chrono 20-25 min.**

## Contexte produit à clarifier en premier

Ce code déclenche des actions **avec effets de bord réels et coûteux** : créer un compte, activer une facturation (= engager de l'argent), envoyer un email à un vrai client. Un bon candidat commence par : "Est-ce que ces actions sont réversibles ? Qui valide avant que l'agent facture réellement un client ? Que se passe-t-il si l'agent se trompe de plan tarifaire ?" C'est exactement le réflexe business/produit attendu — un agent qui orchestre des APIs de prod n'est pas qu'un problème d'ingénierie de prompt, c'est un problème de gouvernance d'actions.

## Bugs / correctness

- `except Exception: result = {}` : si `activate_billing` échoue (ex: le plan "Enterprise" n'existe pas, ou l'API billing est down), l'agent avale l'erreur, la remplace par un dict vide, et continue comme si de rien n'était — il va probablement dire à l'oral suivant que tout s'est bien passé alors que le client n'a pas de facturation active. C'est le bug le plus dangereux : un échec silencieux sur une action métier critique.
- Aucune vérification que le `account_id` utilisé par `activate_billing` correspond bien à celui retourné par `create_crm_account` — le LLM pourrait générer un ID halluciné/différent (ici ça marche par coïncidence car le mock est déterministe, mais rien dans le code ne garantit la cohérence entre étapes).

## Typing

- `args = json.loads(call["arguments"])` renvoie un `dict[str, Any]` non validé — aucune garantie que `contact_email` est bien un email, que `plan_name` fait partie des plans valides, etc. Un modèle Pydantic par outil (`CreateCrmAccountArgs`, `ActivateBillingArgs`...) avec validation résoudrait ça et ferait échouer proprement un appel mal formé plutôt que de planter plus loin.
- Les résultats des clients (`crm.create_account`, etc.) sont des dicts bruts sans schéma — pas de `TypedDict`/dataclass pour `CRMAccountResult`, `BillingResult`.
- La fonction `run_agent` n'a aucune annotation de type (entrée, sortie).

## Separation of concerns

- `run_agent` mélange trois responsabilités : la boucle d'orchestration de l'agent (appeler le LLM, gérer les itérations), le **dispatch** des outils (le `if/elif` sur le nom), et la logique métier de chaque outil (ici déléguée aux clients, mais le mapping outil→client est en dur dans la boucle).
- Le dispatch par `if name == "...": elif name == "...":` duplique l'information déjà présente dans `TOOLS` (la liste des schémas). Ajouter un nouvel outil demande de toucher 2 endroits différents (le schéma ET le dispatch) — source classique de bugs d'oubli. Un registre `{"create_crm_account": crm.create_account, ...}` unique, ou un pattern de classe `Tool` avec `.schema` et `.execute()`, éliminerait la duplication.
- Aucune séparation entre "boucle d'agent générique" (qui pourrait servir pour n'importe quel use case) et "outils spécifiques à l'onboarding" — le fichier ne serait pas réutilisable pour un autre agent sans tout réécrire.

## Maintenabilité

- Noms d'outils en strings littérales dispatchées via `if/elif` : aucune protection contre une faute de frappe (`"activate_biling"` planterait silencieusement dans le `else: result = {}` sans lever d'erreur claire). Un `Enum` ou des constantes nommées limiteraient ce risque.
- Pas de logging structuré (uniquement des `print` dans les mocks) — impossible de savoir en prod quel outil a été appelé, avec quels arguments, à quelle heure, pour quel utilisateur/client.
- Le nombre max d'itérations (`3`) est un magic number en dur dans la boucle `for i in range(3)`, sans lien visible avec le nombre d'outils disponibles (ici il colle exactement au nombre d'étapes du scénario testé — mais c'est fragile : un scénario nécessitant 4 étapes échouerait silencieusement avec "Nombre maximum d'itérations atteint").
- Pas de tests : aucun moyen de vérifier que le dispatch fonctionne correctement pour chaque outil sans lancer tout l'agent.

## Scalabilité

- Tous les appels aux APIs (CRM, billing, email) sont **synchrones et séquentiels**, alors que dans ce scénario ils sont indépendants entre les itérations de l'agent uniquement dans la mesure où le LLM les orchestre un par un — mais dans un vrai FDE use case, on a souvent des lots d'onboarding (plusieurs clients à la fois) : là, sans `async`/parallélisation, le traitement d'un batch de 100 onboardings serait strictement séquentiel et lent.
- `history` et `messages` grossissent sans limite à chaque itération et ne sont jamais tronqués ni résumés — sur un agent avec plus d'itérations ou des conversations plus longues (plus réaliste en prod qu'un scénario à 3 étapes fixes), le contexte envoyé au LLM grossirait jusqu'à dépasser la fenêtre de contexte ou faire exploser le coût par requête.
- Aucun timeout sur les appels aux clients CRM/billing/email : un appel qui pend indéfiniment bloque toute la boucle de l'agent.
- Pas de retry/backoff sur les erreurs transitoires des APIs externes (actuellement toute erreur est simplement avalée, cf. bug ci-dessus, ce qui est pire qu'un retry).

## Angle produit / business (le point à mettre en avant à l'oral)

- **Absence totale de garde-fou avant les actions à effet de bord** (créer un compte, activer une facturation réelle, envoyer un email à un vrai contact) : en prod chez un client, ça pose un risque business et de confiance majeur — un agent qui peut "facturer" ou "contacter" des clients sans validation humaine (ou au moins un mode dry-run/confirmation) est un agent qu'aucune entreprise sérieuse ne mettra en prod tel quel.
- **Pas d'idempotence** : si l'agent est relancé (erreur réseau, retry manuel), rien n'empêche de créer un deuxième compte CRM ou de renvoyer un deuxième email de bienvenue au même client — dans un contexte business réel c'est le genre de duplication qui génère des tickets de support et une mauvaise image auprès du client final.
- **Pas d'audit trail** : impossible de reconstituer a posteriori "quelles actions l'agent a effectivement prises, dans quel ordre, avec quel résultat" — c'est pourtant l'information numéro un qu'un client entreprise demande dès qu'un agent autonome touche à ses données (traçabilité, conformité, debug).
- Pas de métrique de coût/latence par outil, alors que c'est ce qui permettrait de dire au client "l'agent traite un onboarding en X secondes pour Y centimes, contre Z minutes manuellement" — le chiffre d'impact business qu'on attend d'un FDE.

## Comment prioriser à l'oral

1. **Bloquant avant prod** : `except: result = {}` qui avale les échecs sur des actions métier critiques, absence de garde-fou humain avant facturation/envoi d'email réel, absence d'idempotence.
2. **Important, à planifier** : dispatch dupliqué (registre unique d'outils), typage des arguments/résultats, audit log structuré.
3. **Nice-to-have** : passage à l'async pour le traitement en batch, gestion de la troncature de l'historique de conversation.
