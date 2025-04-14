import streamlit as st
import sqlite3
import tempfile
import os
import json
from langchain_community.llms import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import JSONLoader
from langchain.embeddings import FakeEmbeddings  # Replacing OpenAIEmbeddings for compatibility

# App Config
st.set_page_config(page_title="SQL Chat", layout="wide")
st.title("💬 SQL Chat with Insight Generator")

# Upload SQLite file
uploaded_file = st.file_uploader("Upload SQLite .db file", type=["db"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
        tmp_file.write(uploaded_file.read())
        db_path = tmp_file.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Detect schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        schema_info[table_name] = [col[1] for col in columns]

    # Show schema scrollable
    with st.expander("📜 Detected Schema"):
        schema_text = "\n".join(f"{t}: {', '.join(c)}" for t, c in schema_info.items())
        st.code(schema_text, language="sql")

    schema_str = json.dumps(schema_info)

    # Load GROQ LLM
    llm = Groq(model="mixtral-8x7b-32768", api_key=st.secrets["GROQ_API_KEY"])

    # Prompt templates
    sql_prompt = ChatPromptTemplate.from_template("""
        You are an expert in converting natural language questions into SQL queries. 
        Given a database schema: {schema}
        Generate a valid SQLite SQL query for the question: {question}
    """)
    sql_chain = sql_prompt | llm | StrOutputParser()

    insight_prompt = ChatPromptTemplate.from_template("""
        Given the user's question and the SQL result: 
        Question: {question}
        SQL Result: {data}
        Generate a meaningful and helpful business insight.
    """)
    insight_chain = insight_prompt | llm | StrOutputParser()

    # RAG Data
    pairs = [
        {"query": "List all customers", "sql": "SELECT * FROM customers;"},
        {"query": "Show all rental details", "sql": "SELECT * FROM rentals;"},
        {"query": "How many rentals were made?", "sql": "SELECT COUNT(*) FROM rentals;"},
        {"query": "What are the top 5 rented films?", "sql": """
            SELECT film_id, COUNT(*) as rental_count 
            FROM rentals 
            GROUP BY film_id 
            ORDER BY rental_count DESC 
            LIMIT 5;
        """},
        {"query": "Show the list of staff members", "sql": "SELECT * FROM staff;"},
        {"query": "What is the total revenue?", "sql": "SELECT SUM(amount) FROM payments;"},
        {"query": "List top customers by payment", "sql": """
            SELECT customer_id, SUM(amount) as total_paid 
            FROM payments 
            GROUP BY customer_id 
            ORDER BY total_paid DESC 
            LIMIT 5;
        """},
    ]

    rag_path = os.path.join(tempfile.gettempdir(), "faq_rag.json")
    with open(rag_path, "w") as f:
        json.dump(pairs, f)

    loader = JSONLoader(file_path=rag_path, jq_schema=".[]", text_content=False)
    docs = loader.load()

    vectorstore = FAISS.from_documents(docs, FakeEmbeddings())
    retriever = vectorstore.as_retriever()

    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Chat Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Question Loop
    with st.form(key="chat_form"):
        user_input = st.text_input("Ask your question", key="input_text")
        submit = st.form_submit_button("Submit")

    if submit and user_input:
        try:
            # SQL Generation
            sql_query = sql_chain.invoke({"schema": schema_str, "question": user_input})

            # SQL Execution
            cursor.execute(sql_query)
            result = cursor.fetchall()

            # Insight or RAG fallback
            if not result:
                rag_result = rag_chain.run(user_input)
                st.warning("⚠️ No SQL data. Showing fallback from RAG:")
                st.write(rag_result)
                st.session_state.chat_history.append((user_input, rag_result))
            else:
                insight = insight_chain.invoke({
                    "question": user_input,
                    "data": str(result)
                })

                st.success("✅ SQL Query")
                st.code(sql_query.strip(), language="sql")

                st.success("📊 Insight")
                st.write(insight)

                st.session_state.chat_history.append((user_input, insight))

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # Show Chat History
    if st.session_state.chat_history:
        with st.expander("🗂 Chat History"):
            for q, a in st.session_state.chat_history:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown("---")

    # Clean up temp files
    if db_path:
        conn.close()
        os.remove(db_path)