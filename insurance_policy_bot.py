import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph
from typing import TypedDict, Sequence, Annotated
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize LLM
llm = ChatCohere(cohere_api_key=COHERE_API_KEY, model="command-r-plus-08-2024", temperature=0)

# Streamlit UI
st.title("Insurance Policy Information Chatbot")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])

def process_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

retriever = None
vector_store = None
retriever_tool = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        file_path = temp_file.name

    raw_text = process_pdf(file_path) if uploaded_file.type == "application/pdf" else process_txt(file_path)
    st.write(f"Successfully loaded file with {len(raw_text.split())} words.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    st.write(f"Split into {len(chunks)} chunks.")

    embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=COHERE_API_KEY)
    vector_store = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    st.success("Document has been successfully indexed!")

retriever_tool = create_retriever_tool(retriever, "retrieve_docs", "Search and return information about the document uploaded")

query = st.text_input("Enter your query:")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]

def agent(state):
    model = llm.bind_tools([retriever_tool])
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def generate(state):
    prompt = PromptTemplate(template="""
        Context:
        {context}
        Question:
        {question}
        Provide a concise and informative response.
    """, input_variables=["context", "question"])
    
    rag_chain = RunnablePassthrough() | prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": state["messages"][-1].content, "question": state["messages"][0].content})
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()

if st.button("Get Answer"):
    if vector_store and query:
        state = {"messages": [HumanMessage(content=query)]}
        
        st.write("### Debugging Execution Path:")
        
        for output in graph.stream(state):
            for key, value in output.items():
                st.write(f"**Node Reached:** {key}")
                st.write("State:", value)
                st.write("---")
                final_output = value  # Keep updating to get the final state

        if final_output and "messages" in final_output:
            st.subheader("Final Response:")
            st.write(final_output["messages"][-1])
    else:
        st.error("Please upload a file and enter a query!")



