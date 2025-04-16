import os
import sqlite3
import pandas as pd
import streamlit as st
import pickle
import faiss
import traceback
import re

from langchain_groq.chat_models import ChatGroq
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ---- Initialize Components ----

# GROQ API Key Setup
st.sidebar.title("Enter GROQ API Key")
groq_api_key = st.sidebar.text_input("API Key", type="password")
if not groq_api_key:
    st.warning("Please enter your GROQ API key in the sidebar to continue.")

# Initialize Models
llm_sql = ChatGroq(model_name="qwen-2.5-coder-32b", api_key=groq_api_key)
llm_reasoning = ChatGroq(model_name="deepseek-r1-distill-llama-70b", api_key=groq_api_key)

# Initialize Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")
rag_index = None
rag_nl_queries = []
rag_sql_queries = []

# Conversation Memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation_chain = ConversationChain(llm=llm_reasoning, memory=memory, verbose=False)

# ---- Functions ----

# Create RAG Index
def create_rag_index(rag_data_path="rag_db.pkl"):
    global rag_index, rag_nl_queries, rag_sql_queries
    rag_data = [
        # Add your RAG data here
    ]
    rag_nl_queries = [x[0] for x in rag_data]
    rag_sql_queries = [x[1] for x in rag_data]
    embeddings = embedder.encode(rag_nl_queries, show_progress_bar=True)
    rag_index = faiss.IndexFlatL2(embeddings.shape[1])
    rag_index.add(embeddings)

# Analyze Database Schema
def analyze_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = [f"{col[1]} ({col[2]})" for col in cursor.fetchall()]
        cursor.execute(f"PRAGMA foreign_key_list({table[0]})")
        foreign_keys = cursor.fetchall()
        schema[table[0]] = {'columns': columns, 'foreign_keys': foreign_keys}
    conn.close()
    return schema

# Generate SQL Query
def generate_sql_query(schema, user_question, llm_sql):
    schema_lines = [f"# Table: {table}" for table, data in schema.items()]
    for data in schema.values():
        schema_lines += [f"#   - {col}" for col in data['columns']]
    schema_str = "\n".join(schema_lines)
    prompt = f"""# Task: Convert the user's query into a valid SQLite query using this schema:
{schema_str}
# User Question:
{user_question}
# SQL Query:"""
    try:
        response = llm_sql.invoke(prompt)
        return response.content.strip(), "llm"
    except Exception:
        try:
            query_embedding = embedder.encode([user_question])
            _, I = rag_index.search(query_embedding, k=1)
            return rag_sql_queries[I[0][0]], "rag"
        except Exception:
            return "", "error"

# Execute SQL Query
def execute_sql_query(db_path, user_question, schema, llm_sql):
    sql_query, source = generate_sql_query(schema, user_question, llm_sql)
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df, sql_query, None
    except Exception as e:
        return pd.DataFrame(), sql_query, str(e)

# Generate Insights
def generate_insights_from_data(df, user_query):
    if df.empty:
        return "No data returned for analysis."
    preview = df.sample(min(100, len(df)), random_state=42).to_markdown(index=False)
    prompt = f"""
### User Question:
{user_query}
### Data Preview:
{preview}
# Analyze this data and provide insights.
"""
    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(preview)
    response = conversation_chain.invoke({"input": prompt})
    return response["response"]

# Remove THINK Tags
def remove_think_tags(text):
    return re.sub(r"<think>*?</think>", "", text, flags=re.DOTALL)

# ---- Streamlit UI ----

# File Upload
st.sidebar.title("Upload SQLite DB")
uploaded_file = st.sidebar.file_uploader("Choose a SQLite DB file", type=["db"])

if uploaded_file:
    db_path = "uploaded.db"
    with open(db_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Database uploaded successfully.")

    schema = analyze_db_schema(db_path)
    nl_query = st.text_input("Ask a natural language query:")
    if nl_query:
        df, sql_query, error = execute_sql_query(db_path, nl_query, schema, llm_sql)
        st.code(sql_query, language="sql")
        if error:
            st.error(error)
        else:
            st.dataframe(df)
            with st.expander("Get Insights"):
                insight_question = st.text_input("What insights would you like?")
                if insight_question:
                    insights = generate_insights_from_data(df, insight_question)
                    st.markdown(remove_think_tags(insights))
