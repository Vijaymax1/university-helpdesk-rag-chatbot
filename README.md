# 🎓 CampusAssist: RAG-Based University Helpdesk Support Chatbot with Conversational Memory

**CSE 476 Project** — An intelligent, grounded, multi-turn Retrieval-Augmented Generation (RAG) assistant designed for a **University Helpdesk** support role.

---

## 🌟 Key Features

1. **Specific Support Role: University Helpdesk**
   - Curated knowledge base covering 16 realistic campus support areas: Campus Wi-Fi (`eduroam` & Guest), CampusCard Student ID replacement, Duo Multi-Factor Authentication (MFA), Add/Drop course registration, Tuition billing & installment plans, Library book loan periods & renewals, Printing quota & WebPrint, Student Health Center appointments & insurance waivers, Dorm maintenance work orders, Parking permits, Academic advising, Career Center services, Computer lab access & software, Locker rentals, Official transcripts, and Lost & Found.

2. **RAG (Hybrid Semantic & Keyword Retrieval)**
   - **Dense Semantic Retrieval**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) with cosine similarity. Gracefully falls back to normalized TF-IDF vector space if offline.
   - **Keyword Retrieval**: Utilizes BM25 (`rank-bm25`) for exact term matching and acronyms (`eduroam`, `MFA`, `Wi-Fi`, `ID card`).
   - **Hybrid Fusion**: Linear combination of dense and keyword scores.

3. **Conversational Memory & Contextual Query Reformulation**
   - Retains conversation history across dialogue turns in a sliding memory buffer.
   - Resolves pronouns and elliptical fragments (e.g., *"What about guests?"*, *"How much does that cost?"*, *"Where do I pick it up?"*) by automatically reformulating them into disambiguated standalone search queries.

4. **Strict Grounding & Hallucination Prevention**
   - Implements a confidence threshold (default `0.45`).
   - If a query is unrelated or outside the knowledge base (e.g. *"What is the recipe for chocolate cake?"* or *"What is the capital of France?"*), the system recognizes the knowledge is unavailable and outputs an official disclaimer directing the user to helpdesk contacts:
     > *"I apologize, but that information is not available in the University Helpdesk knowledge base. For further assistance, please contact the IT & Campus Helpdesk directly at helpdesk@university.edu or call (555) 019-2834."*

5. **Multi-LLM & Built-in Offline Local Engine**
   - **Groq API**: High-speed inference using `qwen/qwen3.8-27b` or `llama` models via Groq API key.
   - **Built-in Local Grounded Engine**: 100% offline, zero-API-key grounded synthesizer for immediate testing without internet access or paid keys.
   - Optional OpenAI and Google Gemini support.

6. **Interactive Interfaces**
   - **Streamlit Web Application (`app.py`)**: Full-featured chat UI with source expanders, memory inspector, confidence scores, and knowledge base browser.
   - **Jupyter Notebook (`rag_chatbot_cse476.ipynb`)**: Complete walkthrough of each component.
   - **CLI Tool (`cli.py`)**: Interactive terminal loop and `--test` automated suite.

---

## 📁 Repository Structure

```
chatbotcse476/
├── data/
│   └── university_faq.json          # 16 Curated Q&A pairs with metadata & keywords
├── src/
│   ├── __init__.py
│   ├── knowledge_base.py            # Knowledge base loader, validator & chunker
│   ├── retriever.py                 # Hybrid dense/BM25 retriever & threshold gating
│   ├── memory.py                    # Conversational memory & query reformulation
│   ├── generator.py                 # Grounded generator (Groq, OpenAI, Gemini, Local)
│   └── rag_pipeline.py              # Main RAG pipeline coordinator
├── app.py                           # Interactive Streamlit Web Chat Interface
├── cli.py                           # Terminal chat interface & automated tester
├── rag_chatbot_cse476.ipynb         # Interactive Jupyter Notebook demonstration
├── tests/
│   ├── __init__.py
│   └── test_rag.py                  # Automated Pytest unit and integration test suite
├── .env                             # Environment variables (GROQ_API_KEY)
├── .gitignore                       # Git ignore file for secrets and caches
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation and architecture guide
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment
All required packages (`streamlit`, `sentence-transformers`, `groq`, `scikit-learn`, `pytest`, etc.) run directly with your system Python (Python 3.10+). **No virtual environment (`venv`) is required.**

### 2. Optional Dependency Check
If running on a fresh machine:
```bash
pip install -r requirements.txt
```

### 3. API Key Setup
If you have a Groq API Key, create or edit the `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Note: If no API key is provided, the chatbot will automatically run using its built-in offline Local Grounded Engine!)*

---

## 🖥️ Running the Application

### Option A: Interactive Streamlit Web Interface
Launch the Streamlit app directly:
```bash
python -m streamlit run app.py
```
*(or `streamlit run app.py` if scripts are on your PATH)*
- Open your browser at `http://localhost:8501`.
- Explore quick starter question buttons, view retrieved source citations, adjust the relevance threshold, or browse the complete FAQ database in the sidebar!

### Option B: Interactive Terminal (CLI)
Start the command-line chat:
```bash
python cli.py
```
To run the automated verification sequence:
```bash
python cli.py --test
```

### Option C: Jupyter Notebook
Launch the notebook to inspect step-by-step pipeline execution:
```bash
jupyter notebook rag_chatbot_cse476.ipynb
```

---

## 🧪 Running Automated Tests

Run the comprehensive pytest suite:
```bash
python -m pytest -v tests/test_rag.py
```
This tests:
- Knowledge base schema integrity and 16 FAQ items
- Hybrid retrieval accuracy on known university topics
- Out-of-domain query rejection below threshold
- Conversational memory follow-up and pronoun detection
- Contextual query reformulation
- Strict grounded generator disclaimers
- End-to-end multi-turn dialogue flow
