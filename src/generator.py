"""
Grounded Response Generator for University Helpdesk RAG.
Supports Groq (e.g. qwen/qwen3.8-27b), OpenAI, Google Gemini, and a built-in Local Grounded Engine.
Strictly grounds answers in retrieved knowledge chunks and disclaims missing information.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from .retriever import RetrievalResult

# Ensure .env is loaded
load_dotenv()

# Standard out-of-domain / missing knowledge fallback message
UNAVAILABLE_DISCLAIMER = (
    "I apologize, but that information is not available in the University Helpdesk knowledge base. "
    "For further assistance, please contact the IT & Campus Helpdesk directly at "
    "helpdesk@university.edu, call (555) 019-2834, or visit Student Center Room 101."
)

SYSTEM_PROMPT = """You are CampusAssist, the official University Helpdesk Support AI Assistant.
Your mission is to provide accurate, helpful, and courteous support to university students, faculty, and visitors.

STRICT GROUNDING RULES:
1. Answer the user's question using ONLY the provided Knowledge Base Context.
2. Do NOT hallucinate, extrapolate, or introduce external facts or policies not mentioned in the context.
3. If the provided context does not contain enough information to answer the question, or if no relevant context is found, you MUST state clearly that the information is unavailable in the university helpdesk knowledge base and refer the user to helpdesk@university.edu or (555) 019-2834.
4. Maintain a polite, professional, and supportive tone.
5. Include relevant contact details, office locations, or portal URLs if present in the context.
6. When relevant, reference previous conversation context to provide coherent follow-up answers.
"""


class LocalGroundedGenerator:
    """
    Built-in local grounded synthesizer that works completely offline with zero API keys.
    Extracts relevant facts directly from retrieved chunks and formats a coherent response.
    """

    def generate(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        is_relevant: bool,
        history_text: Optional[str] = None
    ) -> str:
        if not is_relevant or not retrieved_results:
            return UNAVAILABLE_DISCLAIMER

        top_chunk = retrieved_results[0]
        
        # Build answer strictly from the retrieved chunks
        response_lines = [
            f"**{top_chunk.category} Support:**",
            top_chunk.answer
        ]

        # If there is a highly relevant secondary chunk that adds useful context
        if len(retrieved_results) > 1 and retrieved_results[1].score >= 0.40:
            second_chunk = retrieved_results[1]
            if second_chunk.category != top_chunk.category or second_chunk.id != top_chunk.id:
                response_lines.append(f"\n*Related Note ({second_chunk.category})*: {second_chunk.answer}")

        return "\n\n".join(response_lines)


class GroundedGenerator:
    """
    Unified Grounded Response Generator supporting Groq, OpenAI, Gemini, and Local Fallback.
    """

    def __init__(
        self,
        provider: str = "auto",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        provider: 'groq', 'openai', 'gemini', 'local', or 'auto'
        """
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.local_generator = LocalGroundedGenerator()
        self._init_backend()

    def _init_backend(self):
        """Discovers and initializes the chosen LLM backend."""
        # Auto-detect provider if auto
        if self.provider == "auto":
            if self.api_key or os.getenv("GROQ_API_KEY"):
                self.provider = "groq"
            elif os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                self.provider = "gemini"
            else:
                self.provider = "local"

        if self.provider == "groq":
            key = self.api_key or os.getenv("GROQ_API_KEY")
            if not key:
                self.provider = "local"
            else:
                try:
                    from groq import Groq
                    self.groq_client = Groq(api_key=key)
                    # Default to verified high-performance available model on Groq
                    self.model_name = self.model_name or "qwen/qwen3.8-27b"
                except Exception:
                    self.provider = "local"

        elif self.provider == "openai":
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                self.provider = "local"
            else:
                try:
                    from openai import OpenAI
                    self.openai_client = OpenAI(api_key=key)
                    self.model_name = self.model_name or "gpt-4o-mini"
                except Exception:
                    self.provider = "local"

        elif self.provider == "gemini":
            key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                self.provider = "local"
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=key)
                    self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                except Exception:
                    self.provider = "local"

    def _format_context(self, retrieved_results: List[RetrievalResult]) -> str:
        """Formats retrieved chunks into a clear context prompt block."""
        context_parts = []
        for i, chunk in enumerate(retrieved_results, 1):
            context_parts.append(
                f"[Source Document {i}] (Category: {chunk.category}, Relevance Score: {chunk.score:.2f})\n"
                f"Question: {chunk.question}\n"
                f"Information: {chunk.answer}"
            )
        return "\n\n".join(context_parts)

    def generate(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        is_relevant: bool,
        history_text: Optional[str] = None
    ) -> str:
        """
        Generates a grounded response using the active provider.
        """
        # If knowledge base retrieval determined the query is out-of-domain or below threshold:
        if not is_relevant or not retrieved_results:
            return UNAVAILABLE_DISCLAIMER

        # Format context
        context_str = self._format_context(retrieved_results)

        # Build user prompt
        prompt_parts = []
        if history_text:
            prompt_parts.append(f"Conversation History:\n{history_text}\n")
        
        prompt_parts.append(f"Knowledge Base Context:\n{context_str}\n")
        prompt_parts.append(f"User Query: {query}\n")
        prompt_parts.append(
            "Instructions: Answer the query accurately and politely using ONLY the Knowledge Base Context above. "
            "If the context does not supply the answer, state that the information is unavailable in the knowledge base."
        )
        user_prompt = "\n".join(prompt_parts)

        # Call active provider
        if self.provider == "groq" and hasattr(self, "groq_client"):
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                answer = chat_completion.choices[0].message.content
                if answer and answer.strip():
                    return answer.strip()
                return self.local_generator.generate(query, retrieved_results, is_relevant, history_text)
            except Exception as e:
                # Fallback to local grounded generator on API failure
                print(f"[Warning] Groq API call failed: {e}. Falling back to Local Grounded Generator.")
                return self.local_generator.generate(query, retrieved_results, is_relevant, history_text)

        elif self.provider == "openai" and hasattr(self, "openai_client"):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                answer = response.choices[0].message.content
                if answer and answer.strip():
                    return answer.strip()
                return self.local_generator.generate(query, retrieved_results, is_relevant, history_text)
            except Exception as e:
                print(f"[Warning] OpenAI API call failed: {e}. Falling back to Local Grounded Generator.")
                return self.local_generator.generate(query, retrieved_results, is_relevant, history_text)

        elif self.provider == "gemini" and hasattr(self, "gemini_model"):
            try:
                full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
                response = self.gemini_model.generate_content(full_prompt)
                if response and response.text and response.text.strip():
                    return response.text.strip()
                return self.local_generator.generate(query, retrieved_results, is_relevant, history_text)
            except Exception as e:
                print(f"[Warning] Gemini API call failed: {e}. Falling back to Local Grounded Generator.")
                return self.local_generator.generate(query, retrieved_results, is_relevant, history_text)

        # Default Local Grounded Engine
        return self.local_generator.generate(query, retrieved_results, is_relevant, history_text)

    def reformulate_query(self, query: str, history_text: str) -> Optional[str]:
        """
        Uses an LLM to reformulate a follow-up query into a standalone query.
        Returns None if using local generator or if API call fails.
        """
        if self.provider == "groq" and hasattr(self, "groq_client"):
            try:
                prompt = (
                    f"Conversation History:\n{history_text}\n\n"
                    f"Latest User Query: {query}\n\n"
                    f"Rewrite the latest user query into a single standalone search query that resolves all pronouns "
                    f"and ellipsis. Do NOT answer the query. Return ONLY the rewritten query text."
                )
                res = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=60
                )
                ans = res.choices[0].message.content
                if ans and ans.strip():
                    return ans.strip().strip('"')
            except Exception:
                return None
        return None
