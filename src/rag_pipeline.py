"""
End-to-End RAG Pipeline for University Helpdesk Support Chatbot.
Orchestrates Knowledge Base, Retriever, Conversational Memory, and Grounded Generator.
"""

from typing import Dict, Any, Optional, List
from .knowledge_base import KnowledgeBase
from .retriever import HybridRetriever, RetrievalResult
from .memory import ConversationMemory
from .generator import GroundedGenerator, UNAVAILABLE_DISCLAIMER


class RAGPipeline:
    """
    Main pipeline orchestrating conversational RAG with grounding and memory.
    """

    def __init__(
        self,
        faq_file_path: Optional[str] = None,
        generator_provider: str = "auto",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        relevance_threshold: float = 0.45,
        top_k: int = 3,
        use_dense_embeddings: bool = True
    ):
        self.kb = KnowledgeBase(faq_file_path)
        self.retriever = HybridRetriever(
            chunks=self.kb.get_chunks_for_indexing(),
            use_dense=use_dense_embeddings,
            relevance_threshold=relevance_threshold
        )
        self.memory = ConversationMemory()
        self.generator = GroundedGenerator(
            provider=generator_provider,
            api_key=api_key,
            model_name=model_name
        )
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def set_relevance_threshold(self, threshold: float):
        """Updates the relevance cutoff threshold."""
        self.relevance_threshold = threshold
        self.retriever.relevance_threshold = threshold

    def set_top_k(self, k: int):
        """Updates number of chunks to retrieve."""
        self.top_k = k

    def reset_conversation(self):
        """Clears conversational memory."""
        self.memory.clear()

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the complete RAG cycle for an incoming user query.

        Steps:
        1. Contextual Query Reformulation (if follow-up)
        2. Knowledge Base Retrieval with confidence gating
        3. Grounded Response Generation (strict guardrails)
        4. Memory Update
        """
        user_query = user_query.strip()
        if not user_query:
            return {
                "response": "Please enter a question or topic so I can assist you.",
                "search_query": "",
                "is_followup": False,
                "is_relevant": False,
                "sources": [],
                "confidence_score": 0.0
            }

        # Step 1: Conversational Memory & Coreference Check
        is_followup = self.memory.is_followup_query(user_query)
        if is_followup:
            # Try LLM reformulation first, then fall back to heuristic
            search_query = self.memory.reformulate_query(
                user_query,
                llm_reformulator=self.generator.reformulate_query
            )
        else:
            search_query = user_query

        # Step 2: Knowledge Base Retrieval
        retrieved_results, is_relevant = self.retriever.retrieve(
            query=search_query,
            top_k=self.top_k,
            threshold=self.relevance_threshold
        )

        top_score = retrieved_results[0].score if retrieved_results else 0.0

        # Step 3: Grounded Answer Generation
        history_text = self.memory.get_formatted_history(num_turns=3) if self.memory.history else None
        
        response_text = self.generator.generate(
            query=user_query,
            retrieved_results=retrieved_results,
            is_relevant=is_relevant,
            history_text=history_text
        )

        # Step 4: Update Conversational Memory
        source_ids = [r.id for r in retrieved_results] if is_relevant else []
        self.memory.add_user_message(user_query)
        self.memory.add_assistant_message(response_text, sources=source_ids)

        return {
            "response": response_text,
            "search_query": search_query,
            "is_followup": is_followup,
            "is_relevant": is_relevant,
            "confidence_score": top_score,
            "sources": [r.to_dict() for r in retrieved_results],
            "history_length": len(self.memory.history)
        }
