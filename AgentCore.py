import csv 
import os
from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
from langchain.agents import create_agent

from bedrock_agentcore.runtime import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

load_dotenv()

def load_faq_csv(path: str) -> List[Document]:
    docs=[]
#opens file with context manager. Closes file safely after code execution.
#utf-8 encoding prevents error while dealing with special characters
    with open(path, "r", encoding="utf-8") as f: 

#dictiReader object to read rows as dictionary with header as key insead of list of string
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()     #strip() removes any spaces, tabs
            a = row["answer"].strip()
            docs.append(Document(page_content=f"Q: {q}\nA: {a}")) #saves formatted string in new document obj
        return docs

docs = load_faq_csv("./lauki_qna.csv")
emb = HuggingFaceEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2",
)
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=0)
chunks = splitter.split_documents(docs)
store = FAISS.from_documents(chunks, emb)

@tool
def search(query: str) -> str:
    """Search the FAQ knowledge base for relevant information.
    Use this tool when the user asks questions about products, services, or policies.
    
    Args: 
    Returns:
        Relevant FAQ entries that might answer the question
    """
#similarity_search kya thi ayu?
    results = store.similarity_search(query, k =3)

    if not results:
        return "No relevant FAQs found"

#aa code samjo
    context = "\n\n----\n\n".join(
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    )
    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"

@tool
def search_detailed(query: str, num: int=5) -> str:
    """Search the FAQ knowledge base with more results for complex queries.
    Use this when the initial search doesn't provide enough information.
    
    Args:
        query: The search query
        num_results: Number of results to retrieve (default: 5)
        
    Returns:
        More comprehensive FAQ entries
    """
    results = store.similarity_search(query, k=num)

    if not results:
        return "no relevent FAQ found"

    context ="\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

@tool
def refomulate_query(query: str, focused: str)-> str:
    """Reformulate the query to focus on a specific aspect.
    Use this when you need to search for a different angle of the question.
    
    Args:
        original_query: The original user question
        focus_aspect: The specific aspect to focus on (e.g., "pricing", "activation", "troubleshooting")
        
    Returns:
        A reformulated query focused on the specified aspect
    """
    reformulated = f"{focused} related to {query}"
    results = store.similarity_search(reformulated, k=3)

#checking for empty results.if results will be empty then value will be 0 and it will turn to 1 and condition fullfilled
    if not results:        
        return f"No results found for {focused}"

    context = "\n\n---\n\n".join([
        f"Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])
    return f"Result for '{focused}' aspect:\n \n{context}"

tools = [search, search_detailed, refomulate_query ]

model= ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_Key")      #python can't read env file directly, need to load it into memory at runtime
)

system_prompt = """You are a helpful FAQ assistant with access to a knowledge base.

Your goal is to answer user questions accurately using the available tools.

Guidelines:
1. Start by using the search_faq tool to find relevant information
2. If the initial search doesn't provide enough info, use search_detailed_faq for more results
3. If the query is complex, use reformulate_query to search different aspects
4. Synthesize information from multiple tool calls if needed
5. Always provide a clear, concise answer based on the retrieved information
6. If you cannot find relevant information, clearly state that

Think step-by-step and use tools strategically to provide the best answer."""

agent = create_agent(
    model = model,
    tools = tools,
    system_prompt=system_prompt,
)

@app.entrypoint
def agent_invocation(payload, context):
    """Handler for agent invocation in AgentCore runtime"""
    print("payload: ", payload)
    print("Context: ", context)

    #payload is dict
    query = payload.get("prompt", "No prmopt found in input")
    result = agent.invoke({"messages": [("human", query)]})

    print("result: ", result)

    return {"result": result['messages'][-1].content}


if __name__=="__main__":
    app.run()