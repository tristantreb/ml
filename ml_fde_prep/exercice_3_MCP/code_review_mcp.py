"""
jira_copilot.py

Copilote interne branché sur le serveur MCP "Jira" de l'entreprise : les
ingénieurs posent une question en langage naturel dans Slack ("où en est le
ticket INFRA-482 ?"), le copilote interroge Jira via MCP et répond dans le
thread Slack.
"""

import os

session = None  # session MCP globale, initialisée au premier appel


# --- MOCKS : simule un client MCP et un LLM, pour que ce fichier tourne sans dépendances externes ---
class MockMCPSession:
    def __init__(self, server_url, token):
        print(f"[MCP] Connexion à {server_url}")
        self.server_url = server_url
        self.token = token

    def list_tools(self):
        return [
            {"name": "get_issue", "description": "Récupère un ticket Jira par sa clé"},
            {"name": "search_issues", "description": "Recherche des tickets par mot-clé"},
        ]

    def call_tool(self, name, arguments):
        if name == "get_issue":
            return {"key": arguments["key"], "status": "In Progress", "assignee": "sophie.durand", "summary": "Latence élevée sur le service de paiement"}
        elif name == "search_issues":
            return {"issues": [{"key": "INFRA-482", "status": "In Progress"}]}
        raise ValueError("Outil inconnu")


class MockLLMClient:
    def complete(self, messages):
        question = messages[-1]["content"]
        if "INFRA-" in question:
            key = [w for w in question.split() if w.startswith("INFRA-")][0].strip("?.,")
            return {"tool_call": {"name": "get_issue", "arguments": {"key": key}}}
        return {"tool_call": None, "content": "Je n'ai pas trouvé de ticket correspondant à votre question."}
# --- fin des mocks ---

llm = MockLLMClient()


def get_mcp_session():
    global session
    server_url = os.environ["JIRA_MCP_SERVER_URL"]
    token = os.environ["JIRA_MCP_TOKEN"]
    session = MockMCPSession(server_url, token)
    return session


def handle_slack_question(user_question):
    mcp = get_mcp_session()
    tools = mcp.list_tools()

    messages = [
        {"role": "system", "content": f"Tu as accès aux outils Jira suivants: {tools}"},
        {"role": "user", "content": user_question},
    ]

    response = llm.complete(messages)

    if response["tool_call"] is not None:
        tool_name = response["tool_call"]["name"]
        tool_args = response["tool_call"]["arguments"]
        result = mcp.call_tool(tool_name, tool_args)

        text = "Ticket " + result["key"] + " : " + result["status"] + " (assigné à " + result["assignee"] + ")\n" + result["summary"]
    else:
        text = response["content"]

    slack_message = {"text": text, "channel": "#eng-support"}
    print("Envoi Slack:", slack_message)
    return slack_message


if __name__ == "__main__":
    os.environ.setdefault("JIRA_MCP_SERVER_URL", "https://mcp.internal.acme.com/jira")
    os.environ.setdefault("JIRA_MCP_TOKEN", "dummy-token-for-practice")
    handle_slack_question("où en est le ticket INFRA-482 ?")
