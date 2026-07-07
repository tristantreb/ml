# Préparation entretien code review — Mistral (ML Forward Deployed Engineer)

## Contenu

- `00_contexte_ML_FDE.md` — contexte du rôle, cycle de vie d'un projet FDE, méthode de review à l'oral.
- `exercice_1_RAG/` — bot de support client avec RAG.
- `exercice_2_agentique/` — agent d'onboarding orchestrant CRM/billing/email.
- `exercice_3_MCP/` — copilote interne connecté à Jira via MCP.

Chaque exercice = 1 fichier `.py` réel et exécutable (LLM et APIs externes mockés pour tourner sans clé API) + 1 fichier `correction_*.md` séparé.

## Méthode d'entraînement recommandée

1. Lis `00_contexte_ML_FDE.md` une fois, pour avoir le cadre en tête (surtout la partie méthode : contexte produit → bugs → design → prod-readiness/business, avec priorisation finale).
2. Pour chaque exercice, dans l'ordre RAG → agentique → MCP (difficulté croissante en surface de code) :
   - Ouvre uniquement le fichier `.py`, lance un chrono de 20-25 minutes (durée réaliste d'un exercice de code review en entretien).
   - Fais ta review à voix haute comme si tu étais face à un interviewer : contexte produit d'abord, puis bugs, puis typing/SoC/maintenabilité/scalabilité, puis angle business, puis priorisation (bloquant / important / nice-to-have).
   - Idéalement, enregistre-toi ou note tes remarques par écrit pour comparer objectivement après.
   - Ouvre ensuite le `correction_*.md` correspondant et compare : qu'as-tu trouvé, qu'as-tu manqué, et surtout — as-tu spontanément relié tes remarques techniques à un impact business/produit sans qu'on te le demande ?
3. Exécute le fichier (`python exercice_X/code_review_XXX.py`) si tu veux voir le comportement réel du code avant/après un refactor que tu proposerais.

## Point d'attention n°1 (ton feedback de screening)

Sur les 3 exercices, la correction contient une section "Angle produit / business" — force-toi à formuler ce type de remarque **avant** qu'on te pousse dessus, dès ta première passe de review. C'est le point spécifiquement identifié comme à travailler.

## Vérification technique

Les 3 scripts ont été testés : ils compilent et s'exécutent sans erreur (mocks inclus, aucune clé API requise).
