import sys
import os
import json
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

# Configuration File
with open("./config", "r") as f:
    config_data = json.load(f)

# Memory Management for the OCI Resource Parameters
class MemoryState:
    def __init__(self):
        self.messages = []

# Define the language model
llm = ChatOCIGenAI(
    model_id="cohere.command-r-08-2024",
    service_endpoint=config_data["llm_endpoint"],
    compartment_id=config_data["compartment_id"],
    auth_profile=config_data["oci_profile"],
    model_kwargs={"temperature": 0.1, "top_p": 0.75, "max_tokens": 2000}
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are an OCI Operations Agent with access to MCP tools (server `oci-ops`).
        Your goal is to provision and manage OCI resources **without requiring the user to know OCIDs**.
                
        INTERACTION RULES:
        1) Wait until the user ask to create a resource
        2) If all the parameters has the ocid information, create the resource
        3) If all the parameters were filled by the user, create the resource
        4) If a parameter given is a name and needs to be converted to a OCID, search for it automatically
        5) If a parameter is missing, ask for the information
        6) Do not wait for a response from creation. Inform "Creation of resource is Done."
        
        IMPORTANT RULES:
        1) Never invent OCIDs. Prefer to ask succinct follow-ups.
        2) Prefer to reuse defaults from memory when appropriate
        
        OUTPUT STYLE:
        - Questions: short, one parameter at a time.
        - Show: mini-summary with final values.
        - Candidate lists: numbered, with name (type) — ocid — score when available.
    """),
    ("placeholder", "{messages}")
])

# Run the client with the MCP server
async def main():
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

    tools = await client.get_tools()
    if not tools:
        print("❌ No MCP tools were loaded. Please check if the server is running.")
        return

    print("🛠️ Loaded tools:", [t.name for t in tools])

    # Creating the LangGraph agent with in-memory state
    memory_state = MemoryState()
    memory_state.messages = []

    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
    )

    print("🤖 READY")
    while True:
        query = input("You: ")
        if query.lower() in ["quit", "exit"]:
            break
        if not query.strip():
            continue

        memory_state.messages.append(HumanMessage(content=query))
        try:
            result = await agent_executor.ainvoke({"messages": memory_state.messages})
            new_messages = result.get("messages", [])

            # Store new messages
            memory_state.messages.extend(new_messages)

            print("Assist:", new_messages[-1].content)

            formatted_messages = prompt.format_messages()

        except Exception as e:
            print("Error:", e)

# Run the agent with asyncio
if __name__ == "__main__":
    asyncio.run(main())