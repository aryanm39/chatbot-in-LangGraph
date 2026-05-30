from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg 
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from tool import tools
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_core.tools import tool

# llm = ChatGoogleGenerativeAI(
#     model="gemini-flash-latest",
#     temperature=0,
#     google_api_key=os.getenv("GEMINI_API_KEY"),
# )
from langchain_groq import ChatGroq
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm_with_tools = llm.bind_tools(tools)

def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

DATABASE_URL = "postgresql://postgres:0101@localhost:5432/postgres"
connection = psycopg.connect(DATABASE_URL,autocommit=True)
checkpointer = PostgresSaver(conn=connection)
checkpointer.setup()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')
# graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
