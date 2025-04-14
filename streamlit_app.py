import os
import sqlite3
import pandas as pd
import streamlit as st
import pickle
import faiss
import traceback

from langchain_groq.chat_models import ChatGroq
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.schema import HumanMessage

# Load API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize models
llm_sql = ChatGroq(model_name="qwen-2.5-coder-32b", api_key=groq_api_key)
llm_reasoning = ChatGroq(model_name="deepseek-r1-distill-llama-70b", api_key=groq_api_key)

# Initialize embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")
rag_index = None
rag_nl_queries = []
rag_sql_queries = []

# Sidebar - File uploader
st.sidebar.title("📁 Upload SQLite DB")
uploaded_file = st.sidebar.file_uploader("Choose a SQLite DB file", type=["db"])

# RAG Setup
def create_rag_index(rag_data_path: str = "rag_db.pkl"):
    global rag_index, rag_nl_queries, rag_sql_queries

    rag_data = [
        ("Get the first and last name of customers living in Canada.", "SELECT first_name, last_name FROM customer c JOIN address a ON c.address_id = a.address_id JOIN city ci ON a.city_id = ci.city_id JOIN country co ON ci.country_id = co.country_id WHERE co.country = 'Canada';"),
        ("Find all films with a rental rate greater than $4.", "SELECT title, rental_rate FROM film WHERE rental_rate > 4;"),
        ("Show the monthly sales trend over time.", "SELECT strftime('%Y-%m', payment_date) AS month, SUM(amount) AS monthly_sales FROM payment GROUP BY month ORDER BY month;"),
        ("Show how many customers are assigned to each store.", "SELECT store_id, COUNT(customer_id) AS customer_count FROM customer GROUP BY store_id;"),
        ("Get the address and contact details for all stores.", "SELECT s.store_id, a.address, a.phone, c.city, co.country FROM store s JOIN address a ON s.address_id = a.address_id JOIN city c ON a.city_id = c.city_id JOIN country co ON c.country_id = co.country_id;"),
        ("Show each customer's latest rental and the one just before that.", "SELECT customer_id, rental_id, rental_date, LAG(rental_date) OVER (PARTITION BY customer_id ORDER BY rental_date) AS previous_rental_date FROM rental;"),
        ("What are the top 3 rented films per category based on total revenue?", "SELECT category_name, title, revenue FROM (SELECT c.name AS category_name, f.title, SUM(p.amount) AS revenue, RANK() OVER (PARTITION BY c.name ORDER BY SUM(p.amount) DESC) AS rank FROM category c JOIN film_category fc ON c.category_id = fc.category_id JOIN film f ON fc.film_id = f.film_id JOIN inventory i ON f.film_id = i.film_id JOIN rental r ON i.inventory_id = r.inventory_id JOIN payment p ON r.rental_id = p.rental_id GROUP BY c.name, f.title) ranked WHERE rank <= 3 ORDER BY category_name, rank;"),
        ("List the top 5 categories with the highest number of films.", "SELECT cat.name AS category_name, COUNT(fc.film_id) AS film_count FROM category cat JOIN film_category fc ON cat.category_id = fc.category_id GROUP BY cat.category_id ORDER BY film_count DESC LIMIT 5;"),
        ("What are the top 5 most rented films and how many times has each been rented?", "SELECT f.title, COUNT(r.rental_id) AS rental_count FROM film f JOIN inventory i ON f.film_id = i.film_id JOIN rental r ON i.inventory_id = r.inventory_id GROUP BY f.film_id ORDER BY rental_count DESC LIMIT 5;"),
        ("Which actors have acted in the most number of films, and how many films have they acted in?", "SELECT a.actor_id, a.first_name || ' ' || a.last_name AS actor_name, COUNT(fa.film_id) AS film_count FROM actor a JOIN film_actor fa ON a.actor_id = fa.actor_id GROUP BY a.actor_id ORDER BY film_count DESC LIMIT 10;"),
    ]

    rag_nl_queries = [x[0] for x in rag_data]
    rag_sql_queries = [x[1] for x in rag_data]

    embeddings = embedder.encode(rag_nl_queries, show_progress_bar=True)
    rag_index = faiss.IndexFlatL2(embeddings.shape[1])
    rag_index.add(embeddings)

create_rag_index()

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

def generate_sql_query(schema, user_question, llm_sql):
    schema_lines = []
    for table, data in schema.items():
        schema_lines.append(f"# Table: {table}")
        for col in data['columns']:
            schema_lines.append(f"#   - {col}")
    schema_str = "\n".join(schema_lines)

    prompt = f"""# Task: Convert user's natural language query into valid SQLite SQL.
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
            _, I = rag_index.search(query_embedding, k=1)
            return rag_sql_queries[I[0][0]], "rag"
        except:
            return "", "error"

def execute_sql_query(db_path, user_question, schema, llm_sql):
    sql_query, source = generate_sql_query(schema, user_question, llm_sql)
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df, sql_query, None
    except Exception as e:
        return pd.DataFrame(), sql_query, str(e)

def generate_insights_from_data(df, user_query):
    if df.empty:
        return "No data was returned from the query to analyze."
    preview = df.head(20).to_markdown(index=False)

    prompt = f"""You are a strategic analyst. Analyze the data below.

User's Question:
{user_query}

Data:
{preview}

Insights:"""

    response = llm_reasoning.invoke(prompt)
    return response.content.strip()

# Streamlit UI
if uploaded_file is not None:
    db_path = "uploaded.db"
    with open(db_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Database uploaded successfully.")
    schema = analyze_db_schema(db_path)

    nl_query = st.text_input("💬 Ask a natural language query about your data:")
    if nl_query:
        df, sql_query, error = execute_sql_query(db_path, nl_query, schema, llm_sql)
        st.code(sql_query, language="sql")
        if error:
            st.error(error)
        else:
            st.dataframe(df)

            with st.expander("🔍 Get insights from the result"):
                insight_question = st.text_input("What do you want to know from this data?")
                if insight_question:
                    insights = generate_insights_from_data(df, insight_question)
                    st.markdown(insights)
