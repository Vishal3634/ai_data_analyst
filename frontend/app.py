import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Backend URL ──────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: #1e1e2e;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    .insight-box {
        background: #1e1e2e;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #ffffff !important;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "file_info" not in st.session_state:
    st.session_state.file_info = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Dataset Upload")
    uploaded = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files to analyze"
    )

    if uploaded:
        if st.button("🚀 Upload & Analyze", use_container_width=True):
            with st.spinner("Uploading datasets..."):
                try:
                    files = [
                        ("files", (f.name, f.getvalue(), "text/csv"))
                        for f in uploaded
                    ]
                    params = {}
                    if st.session_state.session_id:
                        params["session_id"] = st.session_state.session_id

                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    data = response.json()

                    st.session_state.session_id     = data["session_id"]
                    st.session_state.uploaded_files = data["uploaded_files"]

                    info_resp = requests.get(
                        f"{BACKEND_URL}/files/{data['session_id']}",
                        timeout=10
                    )
                    st.session_state.file_info = info_resp.json().get("files", {})
                    st.success(f"✅ {len(data['uploaded_files'])} file(s) uploaded!")

                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

    if st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown("### 📊 Loaded Datasets")
        for fname in st.session_state.uploaded_files:
            info = st.session_state.file_info.get(fname, {})
            with st.expander(f"📄 {fname}"):
                if info:
                    st.write(f"Rows: {info.get('rows', 'N/A')}")
                    st.write(f"Columns: {len(info.get('columns', []))}")

    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    examples = [
        "What is the average price?",
        "Which city has highest sales?",
        "Show sales by region as bar chart",
        "What is the total revenue?",
        "Show distribution of prices"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.example_query = ex

# ── Main Content ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Data Analyst Platform</h1>
    <p>Upload your CSV data and ask questions in plain English</p>
    <p><small>Powered by Groq (LLaMA 3) + LangChain + LLMOps</small></p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat with Data",
    "📊 Data Preview",
    "🧠 AI Insights",
    "📈 LLMOps Status"
])

# ── Tab 1: Chat ───────────────────────────────────────────────
with tab1:
    if not st.session_state.session_id:
        st.info("👈 Upload a CSV file from the sidebar to get started!")
    else:
        selected_files = st.multiselect(
            "Select datasets to query:",
            options=st.session_state.uploaded_files,
            default=st.session_state.uploaded_files
        )

        default_q = getattr(st.session_state, "example_query", "")
        question = st.text_input("Your question:", value=default_q, placeholder="e.g. What is the total revenue?")

        col1, col2 = st.columns([3, 1])
        with col1:
            ask = st.button("🔮 Ask AI", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if ask and question:
            with st.spinner("🤔 AI is thinking..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/query",
                        json={
                            "session_id": st.session_state.session_id,
                            "question": question,
                            "selected_files": selected_files
                        },
                        timeout=120
                    )
                    result = resp.json()
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result.get("answer", ""),
                        "chart": result.get("chart"),
                        "query_type": result.get("query_type", "")
                    })
                    st.session_state.query_count += 1
                    if hasattr(st.session_state, "example_query"):
                        del st.session_state.example_query
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

        # Chat history
        for item in reversed(st.session_state.chat_history):
            st.markdown(f"**🧑 You:** {item['question']}")
            st.markdown(f"🔵 Agent used: **{item['query_type'].upper()}**")
            st.markdown(f"""
            <div class="answer-box">
                <b>🤖 AI Answer:</b><br>{item['answer']}
            </div>
            """, unsafe_allow_html=True)
            if item.get("chart"):
                import base64
                from io import BytesIO
                chart_bytes = base64.b64decode(item["chart"])
                st.image(chart_bytes, use_container_width=True)
            st.markdown("---")

# ── Tab 2: Data Preview ───────────────────────────────────────
with tab2:
    if not st.session_state.session_id:
        st.info("👈 Upload a CSV file to preview data!")
    else:
        for fname in st.session_state.uploaded_files:
            st.markdown(f"### 📄 {fname}")
            try:
                prev = requests.get(
                    f"{BACKEND_URL}/preview/{st.session_state.session_id}/{fname}",
                    timeout=10
                ).json()
                import pandas as pd
                df_prev = pd.DataFrame(prev["preview"])
                st.dataframe(df_prev, use_container_width=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", prev["shape"][0])
                col2.metric("Columns", prev["shape"][1])
                col3.metric("Features", len(prev["columns"]))
            except Exception as e:
                st.error(f"Preview failed: {str(e)}")

# ── Tab 3: AI Insights ────────────────────────────────────────
with tab3:
    st.markdown("### 🧠 Automated Business Insights")
    st.write("Let AI automatically analyze your data and find key patterns.")

    if not st.session_state.session_id:
        st.info("👈 Upload a CSV file first!")
    else:
        selected_for_insights = st.multiselect(
            "Select datasets for insights:",
            options=st.session_state.uploaded_files,
            default=st.session_state.uploaded_files,
            key="insights_select"
        )

        if st.button("✨ Generate Insights", use_container_width=True):
            with st.spinner("🧠 AI is analyzing patterns in your data..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/insights",
                        json={
                            "session_id": st.session_state.session_id,
                            "selected_files": selected_for_insights
                        },
                        timeout=120
                    )
                    insights = resp.json().get("insights", [])

                    if insights:
                        st.success(f"✅ Found {len(insights)} insights!")
                        for insight in insights:
                            st.markdown(f"""
                            <div class="insight-box">
                                {insight}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No insights generated. Try with more data.")

                except Exception as e:
                    st.error(f"Insights failed: {str(e)}")

# ── Tab 4: LLMOps ─────────────────────────────────────────────
with tab4:
    st.markdown("### 📈 Session Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Queries Asked", st.session_state.query_count)
    col2.metric("Datasets Loaded", len(st.session_state.uploaded_files))
    col3.metric("Session Active", "✅ Yes" if st.session_state.session_id else "❌ No")

    if st.session_state.chat_history:
        import pandas as pd
        query_types = [q["query_type"] for q in st.session_state.chat_history]
        type_counts = pd.Series(query_types).value_counts()
        st.bar_chart(type_counts)

    st.markdown("---")
    st.markdown("### 🔗 LangSmith Dashboard")
    st.markdown("View full LLM traces, latency and token usage at:")
    st.markdown("**[smith.langchain.com](https://smith.langchain.com)**")