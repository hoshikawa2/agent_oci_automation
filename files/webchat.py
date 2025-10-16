# -*- coding: utf-8 -*-
import os
import sys
import json
import asyncio
from flask import Flask, render_template, request
from markupsafe import Markup
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# ----------------------------
# CONFIG
# ----------------------------
with open("./config", "r") as f:
    config_data = json.load(f)

app = Flask(__name__)

# ----------------------------
# MEMORY STATE
# ----------------------------
class MemoryState:
    def __init__(self):
        self.messages = []
        self.parameters = {
            "compartment_id": None,
            "subnet_id": None,
            "availability_domain": None,
            "image_id": None,
            "shape": None,
            "ocpus": None,
            "memoryInGBs": None,
            "display_name": None,
        }
        self.candidates = {}  # <- novo

memory_state = MemoryState()

def safe_json_extract(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            cleaned = match.group(0).strip()
            # If the JSON comes truncated (missing closing braces)
            if cleaned.count("{") > cleaned.count("}"):
                cleaned += "}" * (cleaned.count("{") - cleaned.count("}"))
                print("⚠️ JSON truncated, added closing brace(s).")
            return json.loads(cleaned)
    except Exception as e:
        print("⚠️ Failed to parse JSON:", e)
    return None

def sanitize_json(text: str):
    """
    Try to fix and parse malformed JSON from LLM responses.
    """
    if not text:
        return None

    # Removes extra spaces/lines
    cleaned = text.strip()

    # If text comes outside of JSON, extract only the object
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    # Remove commas before closing object/array
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

    # Balance brackets []
    open_brackets = cleaned.count('[')
    close_brackets = cleaned.count(']')
    if open_brackets > close_brackets:
        cleaned += "]" * (open_brackets - close_brackets)
    elif close_brackets > open_brackets:
        # remove extra brackets at the end
        cleaned = cleaned.rstrip("]")

        # Balance braces {}
    open_braces = cleaned.count('{')
    close_braces = cleaned.count('}')
    if open_braces > close_braces:
        cleaned += "}" * (open_braces - close_braces)
    elif close_braces > open_braces:
        # removes leftover keys at the end
        cleaned = cleaned.rstrip("}")

    try:
        return json.loads(cleaned)
    except Exception as e:
        print("⚠️ Still invalid after sanitization:", e)
        print("🔎 Sanitized content:", cleaned[:500])
        print("🔎 Original:", text)
        return None

def validate_payload(params):
    """Checks if all mandatory parameters are filled in"""
    required = ["compartment_id", "subnet_id", "availability_domain",
                "image_id", "shape", "ocpus", "memoryInGBs", "display_name"]
    missing = [r for r in required if not params.get(r)]
    return missing

def check_truncation(response: dict):
    """
    VChecks if the response was truncated due to token limit.
    The object returned by OCI Generative AI contains token usage.
    """
    try:
        usage = response.get("usage", {})
        if usage:
            completion_tokens = usage.get("completion_tokens", 0)
            max_allowed = llm.model_kwargs.get("max_tokens", 0)
            if completion_tokens >= max_allowed:
                print("⚠️ Response possibly truncated by max_tokens.")
                return True
    except Exception:
        pass
    return False

# ----------------------------
# LLM
# ----------------------------
llm = ChatOCIGenAI(
    model_id="cohere.command-r-08-2024",
    service_endpoint=config_data["llm_endpoint"],
    compartment_id=config_data["compartment_id"],
    auth_profile=config_data["oci_profile"],
    model_kwargs={"temperature": 0.1, "top_p": 0.75, "max_tokens": 4000}
)

# ----------------------------
# PROMPT
# ----------------------------

system_text = """
You are an **OCI Operations Agent** with access to MCP tools (server `oci-ops`).
Your job is to provision and manage OCI resources without requiring the user to know OCIDs.
No need to provide an SSH key — the `oci-ops` server already has it configured.

====================
## TOOLS
- `create_compute_instance` → Create a new Compute instance
- `resolve_image` / `list_images` → Resolve or list images
- `resolve_shape` / `list_shapes` → Resolve or list shapes
- `find_subnet` → Find subnet
- `find_compartment` → Find compartment
- `find_ad` / `list_availability_domains` → Resolve or list availability domains
- `oci_cli_passthrough` → Run raw OCI CLI (expert use only)
====================

## RULES
- Parameters: compartment_id, subnet_id, availability_domain, image_id, shape, ocpus, memoryInGBs, display_name.
- Use **snake_case** for parameters at all times.
- Only when ALL parameters are resolved → build the `create_compute_instance` payload using **camelCase**.
- If ambiguous (>1 results) → return in "candidates" with this format:
- Always use snake_case for "parameters": compartment_id, subnet_id, availability_domain, image_id, shape, ocpus, memoryInGBs, display_name.
- Only when calling `create_compute_instance`, convert to camelCase: compartmentId, subnetId, availabilityDomain, imageId, displayName, shape, shapeConfig.
- Never mix snake_case and camelCase in the same JSON object.

"candidates": {{
  "image_id": [
    {{ "index": 1, "name": "Oracle-Linux-9.6-2025.09.16-0", "ocid": "ocid1.image.oc1....", "version": "2025.09.16", "score": 0.99 }},
    {{ "index": 2, "name": "Oracle-Linux-9.6-2025.08.31-0", "ocid": "ocid1.image.oc1....", "version": "2025.08.31", "score": 0.97 }}
  ]
}}

- Do not include null/None values in candidates.  
- If no matches → just return "ask".  
- If exactly one → assign directly in "parameters".

## OUTPUT CONTRACT
- While resolving:  
{{
  "parameters": {{ ... }},
  "candidates": {{ ... }},   # only if ambiguous
  "ask": "..."             # only if needed
}}

- When all resolved:  
{{
  "compartmentId": "...",
  "subnetId": "...",
  "availabilityDomain": "...",
  "imageId": "...",
  "displayName": "...",
  "shape": "...",
  "shapeConfig": {{ "ocpus": <number>, "memoryInGBs": <number> }}
}}

Then return:  
{{ "result": "✅ Creation of resource is Done." }}

⚠️ JSON must be strictly valid (RFC8259).  
No markdown, no comments, no truncation, no null placeholders.  
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_text),
    ("placeholder", "{messages}")
])

# ----------------------------
# MCP TOOLS
# ----------------------------
client = MultiServerMCPClient(
    {
        "oci-ops": {
            "command": sys.executable,
            "args": ["server_mcp.py"],
            "transport": "stdio",
            "env": {
                "PATH": os.environ.get("PATH", "") + os.pathsep + os.path.expanduser("~/.local/bin"),
                "OCI_CLI_BIN": config_data["OCI_CLI_BIN"],
                "OCI_CLI_PROFILE": config_data["oci_profile"],
            },
        },
    }
)

async def load_tools():
    tools = await client.get_tools()
    if not tools:
        print("❌ No MCP tools loaded")
    else:
        print("🛠️ Loaded tools:", [t.name for t in tools])
    return tools

tools = asyncio.get_event_loop().run_until_complete(load_tools())

# ----------------------------
# AGENT EXECUTOR
# ----------------------------
agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,
)

# ----------------------------
# FLASK ROUTES
# ----------------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/send", methods=["POST"])
def send():
    user_message = request.form["message"]
    memory_state.messages.append(HumanMessage(content=user_message))
    user_html = f"<div class='message-user'>You: {user_message}</div>"

    try:
        # injeta estado atual na conversa
        params_json = json.dumps({"parameters": memory_state.parameters}, indent=2)
        context_message = AIMessage(content=f"Current known parameters:\n{params_json}")

        result = asyncio.run(agent_executor.ainvoke({
            "messages": memory_state.messages + [context_message]
        }))

        debug_info = ""

        if check_truncation(result):
            debug_info += "\n\n⚠️ Warning: Response truncated by max_tokens limit."

        new_messages = result.get("messages", [])

        if new_messages:
            memory_state.messages.extend(new_messages)
            assistant_reply = new_messages[-1].content

            parsed = safe_json_extract(assistant_reply)
            if not parsed:  # fallback se falhar
                parsed = sanitize_json(assistant_reply)

            if parsed and "parameters" in parsed:
                # atualiza parâmetros
                for k, v in parsed["parameters"].items():
                    if v is not None:
                        memory_state.parameters[k] = v
                print("📌 Current status:", memory_state.parameters)

                missing = validate_payload(memory_state.parameters)
                if not missing:
                    print("✅ All parameters filled in. The agent should now create the VM..")
                else:
                    print("⚠️ Faltando parâmetros:", missing)
                if not missing:
                    debug_info += "\n✅ All parameters filled in. The agent should now create the VM.."
                else:
                    debug_info += f"\n⚠️ Missing parameters: {missing}"

                if missing:
                    auto_followup = f"Please resolve the following missing parameters: {missing}"
                    memory_state.messages.append(HumanMessage(content=auto_followup))

                # adiciona debug_info na resposta enviada ao navegador
                assistant_reply += "\n\n" + debug_info

            # se vieram candidatos
            if parsed and "candidates" in parsed and parsed["candidates"]:
                memory_state.candidates = parsed["candidates"]
                print("🔍 Candidates found:", memory_state.candidates)

                # monta bloco HTML de candidatos
                candidates_html = ""
                for param, items in memory_state.candidates.items():
                    candidates_html += f"<b>Options for {param}:</b><br>"
                    for c in items:
                        line = f"{c.get('index')}. {c.get('name')} — {c.get('ocid')} — v{c.get('version')} — score {c.get('score')}"
                        candidates_html += line + "<br>"

                ask_text = parsed.get("ask", "Choose an index or provide the OCID.")
                assistant_reply = f"{json.dumps({'parameters': memory_state.parameters}, ensure_ascii=False)}<br>{candidates_html}<i>{ask_text}</i>"

            else:
                memory_state.candidates = {}

        else:
            assistant_reply = "⚠️ No response from agent."
    except Exception as e:
        assistant_reply = f"⚠️ Erro: {e}"

    bot_html = f"<div class='message-bot'>Assistant: {assistant_reply}</div>"
    return Markup(user_html + bot_html)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)