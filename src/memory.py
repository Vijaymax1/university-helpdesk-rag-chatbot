"""
Conversational Memory and Contextual Query Reformulation Module.
Maintains multi-turn conversation state and resolves coreferences/pronouns
so that follow-up questions can be accurately retrieved.
"""

import re
import datetime
from typing import List, Dict, Any, Optional


class MessageTurn:
    """Represents a single conversational turn."""
    def __init__(self, role: str, content: str, sources: Optional[List[str]] = None, timestamp: Optional[str] = None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.sources = sources or []
        self.timestamp = timestamp or datetime.datetime.now().strftime("%H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "sources": self.sources,
            "timestamp": self.timestamp,
        }


class ConversationMemory:
    """
    Manages conversational turns and performs contextual query reformulation
    for follow-up questions.
    """

    PRONOUN_PATTERNS = re.compile(
        r"\b(it|its|they|them|their|this|that|these|those|there|here|the same|again|renew it|pay for it|cost of that|where is that|what about)\b",
        re.IGNORECASE
    )

    QUESTION_FRAGMENTS = [
        "how much", "what is the cost", "where", "when", "how do i", "can i", "is there a fee",
        "who do i contact", "what about", "is it possible", "how long"
    ]

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[MessageTurn] = []
        self.last_topic: Optional[str] = None

    def add_user_message(self, content: str):
        """Adds a user message turn."""
        self.history.append(MessageTurn(role="user", content=content))
        self._trim_history()

    def add_assistant_message(self, content: str, sources: Optional[List[str]] = None):
        """Adds an assistant message turn."""
        self.history.append(MessageTurn(role="assistant", content=content, sources=sources))
        self._trim_history()

    def _trim_history(self):
        """Maintains history within the maximum turns limit."""
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def clear(self):
        """Resets conversational memory."""
        self.history.clear()
        self.last_topic = None

    def get_history(self) -> List[MessageTurn]:
        """Returns all messages in memory."""
        return self.history

    def get_formatted_history(self, num_turns: int = 4) -> str:
        """Formats the last N turns as text for LLM prompting."""
        recent_turns = self.history[-(num_turns * 2):]
        formatted = []
        for turn in recent_turns:
            prefix = "User: " if turn.role == "user" else "Assistant: "
            formatted.append(f"{prefix}{turn.content}")
        return "\n".join(formatted)

    def is_followup_query(self, query: str) -> bool:
        """
        Determines whether the given query is a follow-up depending on pronouns,
        elliptical fragments, or brevity.
        """
        if not self.history:
            return False

        q_lower = query.strip().lower()

        # Check for pronouns
        if self.PRONOUN_PATTERNS.search(q_lower):
            return True

        # Check if query is very short and starts with fragment phrases
        word_count = len(q_lower.split())
        if word_count <= 6:
            for frag in self.QUESTION_FRAGMENTS:
                if q_lower.startswith(frag) or frag in q_lower:
                    return True

        return False

    def extract_recent_topic(self) -> str:
        """
        Heuristically extracts the central subject/topic from the most recent
        user-assistant exchange.
        """
        if not self.history:
            return ""

        # Check the last user message and the last assistant message
        recent_user_msgs = [t.content for t in self.history if t.role == "user"]
        recent_bot_msgs = [t.content for t in self.history if t.role == "assistant"]

        if not recent_user_msgs:
            return ""

        last_user = recent_user_msgs[-1]
        last_bot = recent_bot_msgs[-1] if recent_bot_msgs else ""

        # Domain keywords mapping
        keywords_map = {
            "wifi": "university Wi-Fi eduroam network connection",
            "wi-fi": "university Wi-Fi eduroam network connection",
            "eduroam": "university Wi-Fi eduroam connection",
            "guest": "campus guest Wi-Fi access",
            "id card": "student ID card CampusCard replacement",
            "campuscard": "student ID card CampusCard replacement",
            "password": "campus password reset identity Duo MFA",
            "duo": "Duo Multi-Factor Authentication MFA setup",
            "mfa": "Duo Multi-Factor Authentication MFA setup",
            "add": "course add drop registration deadline",
            "drop": "course add drop registration deadline",
            "course": "course add drop registration deadline",
            "tuition": "tuition payment methods and due dates",
            "bill": "tuition payment methods and due dates",
            "library": "library books borrowing loan period and renewal",
            "book": "library books borrowing loan period and renewal",
            "print": "student printing quota webprint",
            "printing": "student printing quota webprint",
            "health": "student health center appointment insurance waiver",
            "doctor": "student health center appointment insurance waiver",
            "insurance": "student health insurance waiver",
            "dorm": "residence hall dorm maintenance request work order",
            "housing": "residence hall dorm maintenance request work order",
            "maintenance": "dorm maintenance request work order",
            "parking": "campus parking permit student parking zones",
            "permit": "campus parking permit student parking zones",
            "advising": "academic advising appointment scheduling",
            "advisor": "academic advising appointment scheduling",
            "career": "career center resume review mock interview",
            "resume": "career center resume review mock interview",
            "lab": "campus computer labs hours and software",
            "software": "computer lab software matlab python",
            "locker": "campus locker rental student center",
            "transcript": "official transcript request degree audit",
            "graduation": "graduation degree audit official transcript",
            "lost": "campus lost and found office claiming items",
            "found": "campus lost and found office claiming items"
        }

        # Check for matching keywords in previous user message
        user_lower = last_user.lower()
        for kw, topic in keywords_map.items():
            if kw in user_lower:
                return topic

        # Check in previous bot message if needed
        bot_lower = last_bot.lower()
        for kw, topic in keywords_map.items():
            if kw in bot_lower:
                return topic

        # Fallback to cleaning the last user query
        clean_user = re.sub(r"[?!.,]", "", last_user)
        return clean_user

    def reformulate_query(self, query: str, llm_reformulator=None) -> str:
        """
        Reformulates an elliptical follow-up query into a complete, standalone
        search query containing the contextual subject.
        """
        if not self.history or not self.is_followup_query(query):
            return query

        # If an LLM-based query reformulator is provided and available, use it
        if llm_reformulator is not None:
            try:
                reformulated = llm_reformulator(query, self.get_formatted_history(num_turns=2))
                if reformulated and len(reformulated.strip()) > 3:
                    return reformulated.strip()
            except Exception:
                pass  # Fall back to heuristic reformulation

        # Heuristic / Rule-based contextual reformulation
        topic = self.extract_recent_topic()
        if not topic:
            return query

        # Combine topic with the new question
        reformulated = f"{topic} {query}".strip()
        return reformulated
