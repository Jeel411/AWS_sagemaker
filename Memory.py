import csv
import os 
from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chat_models import create_agent
from langgraph.store.base import BaseStore

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore
from langchain.agents.middleware import AgentMiddleware,AgentState, ModelRequest, ModelResponse
from dotenv import load_dotenv

load_dotenv()

app = BedrockAgentCoreApp()

#memory configuration
REGION = ""
MEMORY_ID =
GROQ = os.getenv("GROQ_API_Key")

#initializeing memory component
checkpointer = AgentCoreMemorySaver(memory_id=MEMORY_ID)
store = AgentCoreMemoryStore(memory_id=MEMORY_ID)

def load_faq_csv(path:str) -> List[Document]:
    