import streamlit as st
import sqlite3
import pandas as pd
import os
import plotly.express as px

try:
    import google.generativeai as genai
except ImportError:
    st.error("Please run: pip install google-generativeai")
    st.stop()

# ==============================================================================
# CONFIG & STYLING
# ==============================================================================
st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .chat-container { border-radius: 12px; padding: 15px; margin-bottom: 10px; }
    .suggestion-btn { width: 100%; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATABASE CONNECTION & SCHEMA
# ==============================================================================
@st.cache_resource(ttl=3600)
def get_db_connection():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ecommerce_analytics.db')
    if not os.path.exists(db_path):
        db_path = os.path.join(os.getcwd(), 'ecommerce_analytics.db')
    try:
        return sqlite3.connect(db_path, check_same_thread=False)
    except:
        return None


def get_schema_context(conn):
    if not conn: return "No database connected."
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = ""
    for table in tables:
        t_name = table[0]
        schema += f"TABLE: {t_name}\nCOLUMNS: "
        cursor.execute(f"PRAGMA table_info({t_name});")
        columns = cursor.fetchall()
        schema += ", ".join([f"{col[1]} ({col[2]})" for col in columns]) + "\n\n"
    return schema


# ==============================================================================
# GEMINI SQL AGENT LOGIC
# ==============================================================================
def gemini_sql_agent(user_question, conn, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.5-flash')

    schema = get_schema_context(conn)

    sql_prompt = f"""
    You are a senior data analyst. I will ask a question. Reply with ONLY a valid SQLite SQL query. No markdown formatting, no explanations. Just the raw SQL string.
    DATABASE SCHEMA:
    {schema}
    RULES:
    1. Only use SELECT statements. 
    2. Add LIMIT 1000.
    3. Filter to order_status = 'delivered' if applicable.
    QUESTION: {user_question}
    """

    try:
        sql_response = model.generate_content(sql_prompt)
        sql_query = sql_response.text.replace("```sql", "").replace("```", "").strip()

        df = pd.read_sql(sql_query, conn)

        narrative_prompt = f"""
        You are a senior analyst reporting to a CEO. 
        User asked: "{user_question}"
        The database returned this data: {df.head(10).to_string()}

        Write a 2-3 sentence narrative explaining the result. Start with the key number. Add business context. End with a short actionable implication. Do not explain the SQL.
        At the very end, on a new line, write exactly one word indicating the best chart type for this data: 'bar', 'line', 'pie', or 'none'.
        """

        narrative_response = model.generate_content(narrative_prompt)
        full_text = narrative_response.text.strip().split('\n')

        chart_type = full_text[-1].strip().lower()
        narrative = '\n'.join(full_text[:-1])

        return {"sql": sql_query, "data": df, "narrative": narrative, "chart": chart_type}

    except Exception as e:
        return {"error": f"I couldn't process that query. Error details: {str(e)}"}


# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ AI Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Get free key at ai.google.dev")
    st.markdown("[Get free Gemini API key →](https://aistudio.google.com/app/apikey)")

    with st.expander("ℹ️ How this works"):
        st.markdown("""
        1. Translates plain English to SQL using Gemini.
        2. Executes the query securely on your local SQLite DB.
        3. Gemini analyzes the resulting DataFrame.
        4. Returns a business narrative + chart.
        """)

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# PAGE HEADER
# ==============================================================================
st.markdown("""
    <h1 style='margin-bottom: 0;'>🤖 Ask My Data</h1>
    <p style='color: #94a3b8;'>Ask any business question in plain English — I'll query the database, generate charts, and explain what I find.</p>
""", unsafe_allow_html=True)
st.markdown("""
    <span style='background:#1e293b; padding:4px 12px; border-radius:4px; font-size:12px;'>⚡ Powered by Gemini 1.5 Flash</span>
    <span style='background:#1e293b; padding:4px 12px; border-radius:4px; font-size:12px; margin-left:8px;'>📦 100,573 Orders</span>
""", unsafe_allow_html=True)
st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

# ==============================================================================
# CHAT MEMORY & DEMO EXCHANGES
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if not api_key:
    st.warning(
        "💡 **Demo Mode Active** — Enter your Gemini API Key in the sidebar to ask live questions. Below are example interactions.")

    st.chat_message("user", avatar="🧑‍💼").write("Which region has the highest late delivery rate?")
    with st.chat_message("assistant", avatar="🤖"):
        st.write(
            "The **North region** has the highest late delivery rate at **23.4%**, which is more than double the national average. This directly correlates with a 0.8-point lower average review score in that region. Prioritising carrier negotiations for the Norte and Nordeste hubs is the highest-impact logistics intervention available.")
        with st.expander("🔍 View generated SQL"):
            st.code("""SELECT customer_region, 
       SUM(CASE WHEN is_late_delivery=1 THEN 1 ELSE 0 END)*100.0 / COUNT(*) as late_rate 
FROM fact_orders 
WHERE order_status='delivered' 
GROUP BY customer_region ORDER BY late_rate DESC""")

        demo_df = pd.DataFrame({"customer_region": ["North", "Northeast", "South", "Central", "Southeast"],
                                "late_rate": [23.4, 18.2, 12.1, 11.5, 10.2]})
        st.dataframe(demo_df, use_container_width=True)
        st.plotly_chart(px.bar(demo_df, x='customer_region', y='late_rate', title="Late Delivery Rate by Region",
                               template="plotly_dark"), use_container_width=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
            st.write(msg["content"])
            if "sql" in msg:
                with st.expander("🔍 View generated SQL"): st.code(msg["sql"])
            if "data" in msg:
                st.dataframe(msg["data"], use_container_width=True)
            if "fig" in msg:
                st.plotly_chart(msg["fig"], use_container_width=True)

    prompt = st.chat_input("Ask a business question... (e.g., 'What was total revenue in 2018?')")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Querying database and analysing..."):
                conn = get_db_connection()
                res = gemini_sql_agent(prompt, conn, api_key)

                if "error" in res:
                    st.error(res["error"])
                    st.session_state.messages.append({"role": "assistant", "content": res["error"]})
                else:
                    st.write(res["narrative"])
                    with st.expander("🔍 View generated SQL"):
                        st.code(res["sql"])
                    st.dataframe(res["data"], use_container_width=True)

                    fig = None
                    if res["chart"] == 'bar' and len(res["data"].columns) >= 2:
                        fig = px.bar(res["data"], x=res["data"].columns[0], y=res["data"].columns[1],
                                     template="plotly_dark")
                    elif res["chart"] == 'line' and len(res["data"].columns) >= 2:
                        fig = px.line(res["data"], x=res["data"].columns[0], y=res["data"].columns[1],
                                      template="plotly_dark")
                    elif res["chart"] == 'pie' and len(res["data"].columns) >= 2:
                        fig = px.pie(res["data"], names=res["data"].columns[0], values=res["data"].columns[1],
                                     template="plotly_dark")

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res["narrative"],
                        "sql": res["sql"],
                        "data": res["data"],
                        "fig": fig
                    })