"""
CampusAssist: RAG-Based University Helpdesk Support Chatbot with Conversational Memory
CSE 476 Project - Streamlit Web Application
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

from src.rag_pipeline import RAGPipeline
from src.knowledge_base import KnowledgeBase
from src.generator import UNAVAILABLE_DISCLAIMER

# Page Configuration
st.set_page_config(
    page_title="CampusAssist | University Helpdesk RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .role-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .reformulation-badge {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-family: monospace;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 0.6rem 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_knowledge_base():
    """Caches the knowledge base in memory."""
    return KnowledgeBase()


def initialize_session_state():
    """Initializes session state variables for chat history and pipeline."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am **CampusAssist**, your University Helpdesk Support Assistant. 🎓\n\n"
                    "I can help answer questions regarding campus Wi-Fi, student IDs, course registration, "
                    "tuition billing, library loans, printing quotas, health services, dorm maintenance, "
                    "parking permits, advising, computer labs, transcripts, and more!\n\n"
                    "How may I assist you today?"
                ),
                "sources": [],
                "search_query": None,
                "is_followup": False,
                "confidence": 1.0
            }
        ]

    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(
            generator_provider="groq" if os.getenv("GROQ_API_KEY") else "local",
            api_key=os.getenv("GROQ_API_KEY"),
            relevance_threshold=0.45,
            top_k=3
        )


initialize_session_state()
kb = get_knowledge_base()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/university.png", width=110)
    st.markdown("### 🎓 CampusAssist Settings")
    st.markdown("**Role**: University Helpdesk Support")
    st.markdown("---")

    # LLM Provider Configuration
    st.subheader("🤖 Generator Backend")
    default_key = os.getenv("GROQ_API_KEY", "")
    provider_options = ["Groq (Llama / Qwen - Recommended)", "Local Grounded Engine (Offline)", "OpenAI", "Google Gemini"]
    selected_provider_label = st.selectbox(
        "Provider",
        provider_options,
        index=0 if default_key else 1
    )

    provider_map = {
        "Groq (Llama / Qwen - Recommended)": "groq",
        "Local Grounded Engine (Offline)": "local",
        "OpenAI": "openai",
        "Google Gemini": "gemini"
    }
    active_provider = provider_map[selected_provider_label]

    api_key_input = ""
    if active_provider in ["groq", "openai", "gemini"]:
        current_key = default_key if active_provider == "groq" else os.getenv(f"{active_provider.upper()}_API_KEY", "")
        api_key_input = st.text_input(
            f"{active_provider.title()} API Key",
            value=current_key,
            type="password",
            help="API Key is securely stored in session memory only."
        )

    # Sync pipeline provider if changed
    if (st.session_state.rag_pipeline.generator.provider != active_provider or 
        st.session_state.rag_pipeline.generator.api_key != api_key_input):
        st.session_state.rag_pipeline = RAGPipeline(
            generator_provider=active_provider,
            api_key=api_key_input if api_key_input else None,
            relevance_threshold=st.session_state.rag_pipeline.relevance_threshold,
            top_k=st.session_state.rag_pipeline.top_k
        )

    st.markdown("---")
    st.subheader("⚙️ RAG Hyperparameters")
    top_k_val = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=5, value=3, step=1)
    threshold_val = st.slider(
        "Relevance Threshold",
        min_value=0.20,
        max_value=0.75,
        value=0.45,
        step=0.05,
        help="Queries scoring below this threshold trigger the grounded 'information unavailable' response."
    )
    st.session_state.rag_pipeline.set_top_k(top_k_val)
    st.session_state.rag_pipeline.set_relevance_threshold(threshold_val)

    debug_mode = st.checkbox("🔍 Debug Mode (Show Retrieval Details)", value=False)

    st.markdown("---")
    st.subheader("🧠 Conversational Memory")
    turn_count = len(st.session_state.rag_pipeline.memory.history) // 2
    st.caption(f"Active Turns in Memory: **{turn_count}**")

    if st.button("🗑️ Clear Chat & Memory", use_container_width=True):
        st.session_state.rag_pipeline.reset_conversation()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation memory reset. How can I help you?",
                "sources": [],
                "search_query": None,
                "is_followup": False,
                "confidence": 1.0
            }
        ]
        st.rerun()

    st.markdown("---")
    with st.expander("📚 Knowledge Base Explorer (16 FAQs)"):
        search_filter = st.text_input("Filter FAQs by keyword", "")
        for item in kb.get_all():
            if search_filter.lower() in item.question.lower() or search_filter.lower() in item.category.lower() or not search_filter:
                st.markdown(f"**[{item.category}]** {item.question}")
                st.caption(item.answer)
                st.markdown("---")


# --- MAIN CHAT PANEL ---
st.markdown('<div class="main-header">CampusAssist AI</div>', unsafe_allow_html=True)
st.markdown(
    '<span class="role-badge">Support Role: University Helpdesk</span> '
    '<span style="color:#6B7280; font-size:0.9rem;">Powered by RAG with Conversational Memory & Grounded Synthesis</span>',
    unsafe_allow_html=True
)

# Suggested question pills
st.markdown("**Sample Questions:**")
col1, col2, col3, col4 = st.columns(4)
sample_prompt = None

if col1.button("📶 Connect to Wi-Fi", use_container_width=True):
    sample_prompt = "How do I connect to the university Wi-Fi network?"
if col2.button("👥 Guest Wi-Fi (Follow-up)", use_container_width=True):
    sample_prompt = "What about guests who do not have university credentials?"
if col3.button("💳 Lost Student ID", use_container_width=True):
    sample_prompt = "How do I get a replacement student ID card?"
if col4.button("🚫 Out-of-Domain Test", use_container_width=True):
    sample_prompt = "Can you give me a recipe for chocolate cake?"

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show contextual query reformulation badge if present
        if msg.get("is_followup") and msg.get("search_query"):
            st.markdown(
                f'<div class="reformulation-badge">🔄 Reformulated Query: "{msg["search_query"]}"</div>',
                unsafe_allow_html=True
            )

        st.markdown(msg["content"])

        # Show sources if available and relevant
        if msg.get("sources"):
            with st.expander(f"📄 Retrieved Sources ({len(msg['sources'])} chunks) | Top Confidence: {msg.get('confidence', 0.0):.2f}"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**[{src['category']}] {src['question']}**  *(Relevance: {src['score']:.2f})*\n\n"
                        f"{src['answer']}"
                    )
                    st.markdown("---")

        if debug_mode and msg.get("search_query"):
            st.caption(f"🔧 Debug: Raw Search Query = `{msg['search_query']}` | Confidence Score = `{msg.get('confidence', 0.0)}`")

# Chat Input Handler
user_input = st.chat_input("Type your question here...") or sample_prompt

if user_input:
    # Append user turn to UI history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process query through RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving campus knowledge and generating answer..."):
            result = st.session_state.rag_pipeline.process_query(user_input)

            if result["is_followup"]:
                st.markdown(
                    f'<div class="reformulation-badge">🔄 Reformulated Query: "{result["search_query"]}"</div>',
                    unsafe_allow_html=True
                )

            st.markdown(result["response"])

            # Grounding and source attribution
            if result["is_relevant"] and result["sources"]:
                with st.expander(f"📄 Retrieved Sources ({len(result['sources'])} chunks) | Top Confidence: {result['confidence_score']:.2f}"):
                    for src in result["sources"]:
                        st.markdown(
                            f"**[{src['category']}] {src['question']}**  *(Relevance: {src['score']:.2f})*\n\n"
                            f"{src['answer']}"
                        )
                        st.markdown("---")

            if debug_mode:
                st.caption(
                    f"🔧 Debug: Reformulated = `{result['search_query']}` | "
                    f"Is Followup = `{result['is_followup']}` | "
                    f"Is Relevant = `{result['is_relevant']}` | "
                    f"Top Score = `{result['confidence_score']:.4f}`"
                )

    # Save assistant turn to UI history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "sources": result["sources"] if result["is_relevant"] else [],
        "search_query": result["search_query"],
        "is_followup": result["is_followup"],
        "confidence": result["confidence_score"]
    })
