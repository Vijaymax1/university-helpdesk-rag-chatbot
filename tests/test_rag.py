"""
Comprehensive Pytest Test Suite for CampusAssist RAG Chatbot
Tests Knowledge Base, Hybrid Retriever, Memory, Grounding, and Pipeline.
"""

import pytest
from src.knowledge_base import KnowledgeBase
from src.retriever import HybridRetriever
from src.memory import ConversationMemory
from src.generator import GroundedGenerator, UNAVAILABLE_DISCLAIMER
from src.rag_pipeline import RAGPipeline


@pytest.fixture
def kb():
    """Initializes KnowledgeBase."""
    return KnowledgeBase()


@pytest.fixture
def retriever(kb):
    """Initializes HybridRetriever."""
    return HybridRetriever(
        chunks=kb.get_chunks_for_indexing(),
        relevance_threshold=0.45
    )


@pytest.fixture
def pipeline():
    """Initializes RAGPipeline in local/mock mode for fast test execution."""
    return RAGPipeline(
        generator_provider="local",
        relevance_threshold=0.45,
        top_k=3
    )


def test_knowledge_base_loading(kb):
    """Verify KB loads properly and contains 16 valid FAQ items."""
    assert len(kb) == 16, f"Expected 16 FAQ items, got {len(kb)}"
    for item in kb.get_all():
        assert item.id.startswith("faq-"), f"Invalid ID format: {item.id}"
        assert len(item.category) > 0, "Missing category"
        assert len(item.question) > 5, "Question too short"
        assert len(item.answer) > 20, "Answer too short"
        assert len(item.keywords) >= 3, "Keywords too few"
        assert len(item.searchable_text) > 30, "Searchable text chunk too short"


def test_retriever_known_queries(retriever):
    """Verify high retrieval relevance scores for known FAQ topics."""
    queries_to_test = [
        ("How do I connect to campus Wi-Fi?", "faq-01"),
        ("How do I replace a lost student ID card?", "faq-02"),
        ("How do I reset my password with Duo MFA?", "faq-03"),
        ("What is the deadline for dropping classes?", "faq-04"),
        ("How many books can I borrow from the library?", "faq-06"),
        ("How much student printing quota do I get?", "faq-07"),
    ]

    for query, expected_id in queries_to_test:
        results, is_relevant = retriever.retrieve(query, top_k=3)
        assert is_relevant is True, f"Query '{query}' was falsely marked as irrelevant"
        assert len(results) > 0
        top_result = results[0]
        assert top_result.id == expected_id, f"Expected {expected_id} as top hit for '{query}', got {top_result.id}"
        assert top_result.score >= 0.50, f"Expected score >= 0.50, got {top_result.score}"


def test_out_of_domain_rejection(retriever):
    """Verify that unrelated out-of-domain queries fail relevance threshold."""
    unrelated_queries = [
        "What is the capital city of France?",
        "How do I bake chocolate chip cookies?",
        "What was the score of the football game yesterday?",
        "Can you write a poem about quantum mechanics?"
    ]

    for query in unrelated_queries:
        results, is_relevant = retriever.retrieve(query, top_k=3, threshold=0.45)
        top_score = results[0].score if results else 0.0
        assert is_relevant is False, f"Unrelated query '{query}' was incorrectly marked relevant with score {top_score}"


def test_conversational_memory_followup_detection():
    """Verify pronoun and follow-up detection in conversational memory."""
    mem = ConversationMemory(max_turns=5)
    assert mem.is_followup_query("What is campus wifi?") is False

    mem.add_user_message("How do I get a replacement student ID card?")
    mem.add_assistant_message("Visit the Campus Card Office in Room 105. A $20 fee applies.")

    assert mem.is_followup_query("How much does it cost?") is True
    assert mem.is_followup_query("Where is that located?") is True
    assert mem.is_followup_query("Can I pay for it online?") is True
    assert mem.is_followup_query("What about guests?") is True


def test_conversational_query_reformulation():
    """Verify contextual query reformulation bridges topics."""
    mem = ConversationMemory()
    mem.add_user_message("How do I connect to the university Wi-Fi network?")
    mem.add_assistant_message("Log into eduroam with your university email.")

    reformulated = mem.reformulate_query("What about guests who do not have credentials?")
    assert "wi-fi" in reformulated.lower() or "wifi" in reformulated.lower() or "eduroam" in reformulated.lower()


def test_grounded_generator_disclaimer():
    """Verify grounded generator outputs standard disclaimer on missing info."""
    gen = GroundedGenerator(provider="local")
    res = gen.generate(
        query="What is the recipe for cookies?",
        retrieved_results=[],
        is_relevant=False
    )
    assert res == UNAVAILABLE_DISCLAIMER


def test_end_to_end_pipeline_flow(pipeline):
    """Verify full end-to-end multi-turn conversation in RAGPipeline."""
    # Turn 1: Valid initial question
    t1 = pipeline.process_query("What are the borrowing limits for library books?")
    assert t1["is_relevant"] is True
    assert "library" in t1["response"].lower() or "books" in t1["response"].lower()
    assert t1["confidence_score"] >= 0.50

    # Turn 2: Follow-up question using coreference
    t2 = pipeline.process_query("Can I renew them online?")
    assert t2["is_followup"] is True
    assert t2["is_relevant"] is True
    assert "renew" in t2["response"].lower()

    # Turn 3: Out-of-domain question
    t3 = pipeline.process_query("Tell me about rocket propulsion.")
    assert t3["is_relevant"] is False
    assert UNAVAILABLE_DISCLAIMER in t3["response"]
