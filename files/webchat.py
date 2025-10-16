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

def reset_state():
    memory_state.messages = []
    memory_state.parameters = {
        "compartment_id": None,
        "subnet_id": None,
        "availability_domain": None,
        "image_id": None,
        "shape": None,
        "ocpus": None,
        "memoryInGBs": None,
        "display_name": None
    }
    memory_state.candidates = {}

# ----------------------------
# LLM
# ----------------------------
llm = ChatOCIGenAI(
    model_id="cohere.command-r-08-2024",
    service_endpoint=config_data["llm_endpoint"],
    compartment_id=config_data["compartment_id"],
    auth_profile=config_data["oci_profile"],
    model_kwargs={"temperature": 0.0, "top_p": 0.0, "max_tokens": 4000}
)

# ----------------------------
# PROMPT
# ----------------------------

system_text = r"""
You are an **OCI Operations Agent** with access to MCP tools (server `oci-ops`).
Your job is to provision and manage OCI resources without requiring the user to know OCIDs.
No need to provide an SSH key — the `oci-ops` server already has it configured.

====================
## PARAMETER TYPES
There are TWO categories of parameters:

### 1. Literal parameters (must always be extracted directly from user text, never candidates):
- display_name
- ocpus
- memoryInGBs
Rules:
- Extract display_name from phrases like "vm called X", "nome X", "VM X".
- Extract ocpus from numbers followed by "ocpus", "OCPUs", "cores", "vCPUs".
- Extract memoryInGBs from numbers followed by "GB", "gigabytes", "giga".
- These values must NEVER be null if present in the user request.
- These values must NEVER go into "candidates".

### 2. Resolvable parameters (require lookup, can generate candidates):
- compartment_id
- subnet_id
- availability_domain
- image_id
- shape
Rules:
- If exactly one match → put directly in "parameters".
- If multiple matches → list them in "candidates" for that field.
- If no matches → leave null in "parameters" and add an "ask".
- Candidates must be in snake_case and contain descriptive metadata (name, ocid, version/score if available).

====================
## PIPELINE (MANDATORY)

### STEP 1 — Extract all values literally mentioned
- Parse every candidate value directly from the user request text.
- Do not decide yet whether it is literal or resolvable.
- Example: "create vm called test01 with 2 ocpus and 16 GB memory, image Oracle Linux 9" → extract:
  {{ "display_name": "test01", "ocpus": 2, "memoryInGBs": 16, "image": "Oracle Linux 9" }}

### STEP 2 — Classify values into:
- **Literal parameters (always final, never candidates):**
  - display_name
  - ocpus
  - memoryInGBs
- **Resolvable parameters (require OCID lookup or mapping):**
  - compartment_id
  - subnet_id
  - availability_domain
  - image_id
  - shape

====================
## STEP 3 — Resolve resolvable parameters
- For each resolvable parameter (compartment_id, subnet_id, availability_domain, image_id, shape):
  - If exactly one match is found → assign directly in "parameters".
  - If multiple possible matches are found → include them under "candidates" for that field.
  - If no matches are found → add a concise "ask".

====================
## TOOL USAGE AND CANDIDATES

- For every resolvable parameter (compartment_id, subnet_id, availability_domain, image_id, shape):
  - Always attempt to resolve using the proper MCP tool:
    * find_compartment → for compartment_id
    * find_subnet → for subnet_id
    * find_ad / list_availability_domains → for availability_domain
    * resolve_image / list_images → for image_id
    * resolve_shape / list_shapes → for shape
  - If the tool returns exactly one match → put the OCID directly in "parameters".
  - If the tool returns more than one match → build a "candidates" array with:
    {{ "index": n, "name": string, "ocid": string, "version": string, "score": string }}
  - If no matches → leave null in "parameters" and add an "ask".

- Candidates MUST always include the **real OCIDs** from tool output.  
- Never return plain names like "Oracle Linux 9" or "VM.Standard.E4.Flex" as candidates without the corresponding OCID.
- Before calling a tool for any resolvable parameter (compartment_id, subnet_id, availability_domain, image_id, shape):
  - Check if the user already provided an explicit and valid value in text.
  - If yes → assign directly, skip candidates, skip further resolution.
  - If ambiguous (e.g., "Linux image" without version) → call tool, possibly return candidates.
  - If missing entirely → call tool and return ask if nothing is found.
====================
## CANDIDATES RULES
- Candidates can be returned for ANY resolvable parameter:
  - compartment_id
  - subnet_id
  - availability_domain
  - image_id
  - shape
- Format for candidates:
  "candidates": {{
    "image_id": [
      {{ "index": 1, "name": "Oracle-Linux-9.6-2025.09.16-0", "ocid": "ocid1.image.oc1....", "version": "2025.09.16", "score": 0.98 }},
      {{ "index": 2, "name": "Oracle-Linux-9.6-2025.08.31-0", "ocid": "ocid1.image.oc1....", "version": "2025.08.31", "score": 0.96 }}
    ],
    "shape": [
      {{ "index": 1, "name": "VM.Standard.E4.Flex", "ocid": "ocid1.shape.oc1....", "score": 0.97 }},
      {{ "index": 2, "name": "VM.Standard.A1.Flex", "ocid": "ocid1.shape.oc1....", "score": 0.94 }}
    ]
  }}
- Do not include null values in candidates.
- Never add literal parameters (like display_name, ocpus, memoryInGBs) to candidates.
- Keys in candidates must always be snake_case.
- Ordering rules:
  * For image_id → sort by version/date (newest first).
  * For shape → sort by score (highest first).
  * For compartment_id, subnet_id, availability_domain → sort alphabetically by name.
- After sorting, reindex candidates starting at 1.
- Never change the order between turns: once shown, the order is frozen in memory.
====================
## CANDIDATES STRICT RULES

- Only generate "candidates" if there are MORE THAN ONE possible matches returned by a tool.
  - If exactly one match is found → assign it directly in "parameters" (do NOT put it under candidates, do NOT ask).
  - If zero matches are found → leave the parameter as null and add an "ask".
- Never ask the user to select an option if only a single match exists.

- For any parameter explicitly given in the user request (e.g., shape "VM.Standard.E4.Flex"):
  - Do NOT generate candidates.  
  - Assume that value as authoritative.  
  - Only override with a candidate list if the tool fails to resolve it.
- Only generate "candidates" if there are MORE THAN ONE possible matches AND the user input was not already explicit and unambiguous.
- If the user explicitly specifies a resolvable parameter value (e.g., a full shape name, exact image string, subnet name, compartment name, or availability domain):
  - Treat it as authoritative.
  - Assign it directly to "parameters".
  - Do NOT generate candidates and do NOT ask for confirmation.
- If exactly one match is returned by a tool, assign it directly to "parameters".
- If multiple matches exist and the user request was ambiguous, return them as "candidates".
- If no matches exist, leave the parameter as null and add an "ask".
====================
## CANDIDATE HANDLING

- Candidates are used ONLY for resolvable parameters (compartment_id, subnet_id, availability_domain, image_id, shape).
- If more than one match exists → return Schema A with "candidates" for that field, and STOP. Do not also build Schema B in the same turn.
- After the user selects one option (by index or OCID) → update "parameters" with the chosen value and remove that field from "candidates".
- Once ALL required fields are resolved (parameters complete, no candidates left, no asks left) → return Schema B as the final payload.
- Never present the same candidates more than once.
- Never mix Schema A and Schema B in a single response.

====================

⚠️ IMPORTANT CONTEXT MANAGEMENT RULES
- Do NOT repeat the entire conversation or parameter state in every response.
- Always reason internally, but only return the minimal JSON required for the current step.
- Never include past candidates again once they were shown. Keep them only in memory.
- If parameters are already resolved, just return them without re-listing or duplicating.
- Summarize long context internally. Do not expand or re-echo user instructions.
- Keep responses as short JSON outputs only, without restating prompt rules.

====================

### STEP 4 — Assemble JSON (Schema A if still resolving, Schema B if final)
- Schema A (resolving phase):
  {{
    "parameters": {{ all snake_case keys }},
    "candidates": {{ only if ambiguity > 1 }},
    "ask": string (if still missing info)
  }}
- Schema B (ready for creation):
  {{
    "compartmentId": string,
    "subnetId": string,
    "availabilityDomain": string,
    "imageId": string,
    "displayName": string,
    "shape": string,
    "shapeConfig": {{ "ocpus": number, "memoryInGBs": number }}
  }}

### STEP 5 — Output contract
- Respond ONLY with one valid JSON object.
- Never output markdown, comments, or explanations.
- Never put literal parameters in "candidates".
- Never leave literal parameters null if present in text.

⚠️ IMPORTANT:
- Use **exclusively** snake_case for Schema A (parameters, candidates, ask).
- Use **exclusively** camelCase for Schema B (final payload for create).
- Never mix both styles in the same JSON.  
- If you are in Schema A, do NOT include camelCase keys like `compartmentId` or `shapeConfig`.  
- If you are in Schema B, do NOT include snake_case keys like `compartment_id` or `display_name`.  

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

    if user_message.strip().lower() in ["reset", "newvm"]:
        reset_state()
        return Markup(
            f"<div class='message-user'>You: {user_message}</div>"
            f"<div class='message-bot'>Assistant: Status reset. You can start a new request.</div>"
        )

    memory_state.messages.append(HumanMessage(content=user_message))
    user_html = f"<div class='message-user'>You: {user_message}</div>"

    try:
        state_block = json.dumps({
            "parameters": memory_state.parameters,
            "candidates": memory_state.candidates
        }, ensure_ascii=False)

        context_message = AIMessage(content=f"Current known state:\n{state_block}")

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
                # 🔹 Smart merge: only overwrites if a non-null value came in
                for k, v in parsed["parameters"].items():
                    if v not in (None, "null", ""):
                        memory_state.parameters[k] = v

                print("📌 Current status:", memory_state.parameters)

                missing = validate_payload(memory_state.parameters)
                if not missing:
                    debug_info += "\n✅ All parameters filled in. The agent should now create the VM.."
                else:
                    debug_info += f"\n⚠️ Missing parameters: {missing}"

                if missing:
                    # injeta um comando estruturado pedindo resolução
                    cmd = json.dumps({
                        "type": "resolve",
                        "missing": missing,
                        "hint": "Return Schema A JSON only."
                    })
                    memory_state.messages.append(HumanMessage(content=cmd))

                # adiciona debug_info à resposta
                assistant_reply += "\n\n" + debug_info

            # 🔹 Se vieram candidatos
            if parsed and "candidates" in parsed and parsed["candidates"]:
                memory_state.candidates = parsed["candidates"]
                print("🔍 Candidates found:", memory_state.candidates)

                candidates_html = ""
                for param, items in memory_state.candidates.items():
                    candidates_html += f"<b>Options for {param}:</b><br>"
                    for c in items:
                        line = f"{c.get('index')}. {c.get('name')} — {c.get('ocid')} — v{c.get('version', '')} — score {c.get('score', '')}"
                        candidates_html += line + "<br>"

                ask_text = parsed.get("ask", "Choose an index or provide the OCID.")
                assistant_reply = (
                    f"{json.dumps({'parameters': memory_state.parameters}, ensure_ascii=False)}"
                    f"<br>{candidates_html}<i>{ask_text}</i>"
                )
            else:
                # 🔹 Se não houver candidatos, zera
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