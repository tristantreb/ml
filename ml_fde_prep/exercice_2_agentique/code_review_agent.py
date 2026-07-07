"""
onboarding_agent.py

Agent qui automatise l'onboarding d'un nouveau client entreprise : il crée la fiche
CRM, active la facturation, et envoie l'email de bienvenue, en orchestrant plusieurs
APIs internes à partir d'une consigne en langage naturel donnée par un commercial.

Exemple d'utilisation : le commercial tape
"Onboarde la société Acme Corp, contact jean@acme.com, plan Enterprise"
et l'agent exécute les actions nécessaires.
"""

import os
import json

# --- MOCKS des APIs internes et du LLM, pour que ce fichier tourne sans dépendances externes ---
class CRMClient:
    def create_account(self, company_name, contact_email):
        print(f"[CRM] Création du compte pour {company_name} ({contact_email})")
        return {"account_id": "acc_" + company_name.lower().replace(" ", "_"), "status": "created"}


class BillingClient:
    def activate_plan(self, account_id, plan_name):
        print(f"[BILLING] Activation du plan {plan_name} pour {account_id}")
        return {"subscription_id": "sub_" + account_id, "plan": plan_name, "status": "active"}


class EmailClient:
    def send_welcome_email(self, contact_email, account_id):
        print(f"[EMAIL] Envoi de l'email de bienvenue à {contact_email}")
        return {"message_id": "msg_" + account_id, "status": "sent"}


class MockLLMClient:
    """Simule un modèle avec function calling : renvoie une séquence scriptée d'appels d'outils."""
    def __init__(self):
        self._step = 0

    def complete(self, messages, tools):
        self._step += 1
        last_user_msg = messages[0]["content"]
        if self._step == 1:
            return {"tool_calls": [{"name": "create_crm_account", "arguments": '{"company_name": "Acme Corp", "contact_email": "jean@acme.com"}'}]}
        elif self._step == 2:
            return {"tool_calls": [{"name": "activate_billing", "arguments": '{"account_id": "acc_acme_corp", "plan_name": "Enterprise"}'}]}
        elif self._step == 3:
            return {"tool_calls": [{"name": "send_welcome_email", "arguments": '{"contact_email": "jean@acme.com", "account_id": "acc_acme_corp"}'}]}
        else:
            return {"tool_calls": None, "content": f"Onboarding terminé pour la demande: {last_user_msg}"}
# --- fin des mocks ---

crm = CRMClient()
billing = BillingClient()
email = EmailClient()
llm = MockLLMClient()

TOOLS = [
    {"name": "create_crm_account", "description": "Crée un compte CRM", "parameters": {"company_name": "string", "contact_email": "string"}},
    {"name": "activate_billing", "description": "Active un plan de facturation", "parameters": {"account_id": "string", "plan_name": "string"}},
    {"name": "send_welcome_email", "description": "Envoie l'email de bienvenue", "parameters": {"contact_email": "string", "account_id": "string"}},
]


def run_agent(user_request):
    messages = [{"role": "user", "content": user_request}]
    history = []

    for i in range(3):  # nombre max d'itérations de l'agent
        response = llm.complete(messages, TOOLS)

        if response["tool_calls"] is None:
            return response["content"]

        for call in response["tool_calls"]:
            name = call["name"]
            args = json.loads(call["arguments"])
            history.append(call)

            try:
                if name == "create_crm_account":
                    result = crm.create_account(args["company_name"], args["contact_email"])
                elif name == "activate_billing":
                    result = billing.activate_plan(args["account_id"], args["plan_name"])
                elif name == "send_welcome_email":
                    result = email.send_welcome_email(args["contact_email"], args["account_id"])
                else:
                    result = {}
            except Exception:
                result = {}

            messages.append({"role": "assistant", "content": f"Appel de {name} avec {args}"})
            messages.append({"role": "tool", "content": json.dumps(result)})

    return "Nombre maximum d'itérations atteint."


if __name__ == "__main__":
    result = run_agent("Onboarde la société Acme Corp, contact jean@acme.com, plan Enterprise")
    print(result)
