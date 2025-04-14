import streamlit as st
import sqlite3
import pandas as pd
import pickle
import faiss
import traceback
from sentence_transformers import SentenceTransformer
from langchain_groq.chat_models import ChatGroq
import os

# Load API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Load models
llm_sql = ChatGroq(model_name="qwen-2.5-coder-32b", api_key=groq_api_key)
llm_reasoning = ChatGroq(model_name="deepseek-r1-distill-llama-70b", api_key=groq_api_key)

# Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global RAG variables
rag_index = None
rag_nl_queries = []
rag_sql_queries = []

def analyze_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_info = [f"{col[1]} ({col[2]})" for col in columns]
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = cursor.fetchall()
        schema[table_name] = {'columns': col_info, 'foreign_keys': foreign_keys}
    conn.close()
    return schema

def create_rag_index():
    global rag_index, rag_nl_queries, rag_sql_queries
    rag_data = [
        ("Get the first and last name of customers living in Canada.", "SELECT first_name, last_name FROM customer c JOIN address a ON c.address_id = a.address_id JOIN city ci ON a.city_id = ci.city_id JOIN country co ON ci.country_id = co.country_id WHERE co.country = 'Canada';"),
        ("Find all films with a rental rate greater than $4.", "SELECT title, rental_rate FROM film WHERE rental_rate > 4;"),
        # Add more pairs as needed
    ]
    rag_nl_queries = [item[0] for item in rag_data]
    rag_sql_queries = [item[1] for item in rag_data]
    embeddings = embedder.encode(rag_nl_queries, show_progress_bar=True)
    rag_index = faiss.IndexFlatL2(embeddings.shape[1])
    rag_index.add(embeddings)

def generate_sql_query(schema: dict, user_question: str, llm_sql):
    schema_lines = []
    for table, data in schema.items():
        schema_lines.append(f"# Table: {table}")
        for col in data['columns']:
            schema_lines.append(f"#   - {col}")
    schema_str = "\n".join(schema_lines)
    prompt = f"""# Task: Convert the user's natural language query into a valid SQLite SQL query.
# Use only the schema provided below.
# Wrap any SQL keyword or mixed-case column/table name in double quotes (e.g., "To").
# Avoid guessing columns or tables not in the schema.
# Only return the SQL query — no explanation or formatting.
# Use SQLite syntax.
# Avoid JOINs unless explicitly requested.
# return the query without quotation marks.

{schema_str}

# User Question:
# {user_question}

# SQL Query:"""

    try:
        response = llm_sql.invoke(prompt)
        return response.content.strip(), "llm"
    except:
        try:
            query_embedding = embedder.encode([user_question])
            D, I = rag_index.search(query_embedding, k=1)
            fallback_sql = rag_sql_queries[I[0][0]]
            return fallback_sql, "rag"
        except Exception as e:
            traceback.print_exc()
            return "", "error"

def execute_sql_query(db_path: str, user_question: str, schema: str, llm_sql):
    sql_query, source = generate_sql_query(schema, user_question, llm_sql)
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df, sql_query, None
    except:
        return pd.DataFrame(), sql_query, "Execution failed"

def generate_insights_from_data(df: pd.DataFrame, user_query: str) -> str:
    if df.empty:
        return "No data returned from the query."
    preview = df.head(20).to_markdown(index=False)
    prompt = f"""
You are a strategic data analyst AI assistant. Analyze the following SQL query result:

### User's Question:
{user_query}

### DATA PREVIEW:
{preview}

### Provide:
1. Summary
2. Trends or anomalies
3. Correlations
4. Predictions
5. Recommendations
"""
    response = llm_reasoning.invoke(prompt)
    return response.content.strip()

# UI
st.title("🧠 Natural Language to SQL with Insights")
st.markdown("Upload a SQLite `.db` file and ask questions in natural language.")

uploaded_file = st.file_uploader("📁 Upload SQLite .db file", type="db")

if uploaded_file:
    with open("temp.db", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ File uploaded")
    create_rag_index()
    schema = analyze_db_schema("temp.db")
    st.subheader("📚 Schema Detected")
    st.json(schema)

    user_question = st.text_input("💬 Enter your natural language question")
    if user_question:
        result_df, sql_query, err = execute_sql_query("temp.db", user_question, schema, llm_sql)
        st.subheader("🧾 Generated SQL")
        st.code(sql_query, language="sql")
        if err:
            st.error(f"❌ Error: {err}")
        else:
            st.subheader("📊 Result Data")
            st.dataframe(result_df)

            insight_query = st.text_input("🔎 Want insights? Add more context")
            if insight_query:
                st.subheader("💡 AI-Powered Insights")
                insights = generate_insights_from_data(result_df, insight_query)
                st.markdown(insights)
