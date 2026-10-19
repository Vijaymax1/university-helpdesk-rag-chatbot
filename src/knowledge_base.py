"""
Knowledge Base Loader and Processor for University Helpdesk FAQ.
Handles loading, schema validation, and chunk preparation for RAG indexing.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class FAQItem:
    """Represents a single FAQ entry."""
    def __init__(self, id: str, category: str, question: str, answer: str, keywords: Optional[List[str]] = None):
        self.id = id
        self.category = category
        self.question = question
        self.answer = answer
        self.keywords = keywords or []

    @property
    def searchable_text(self) -> str:
        """Formatted text chunk optimized for dense vector and keyword retrieval."""
        kw_str = ", ".join(self.keywords)
        return (
            f"Category: {self.category}\n"
            f"Question: {self.question}\n"
            f"Answer: {self.answer}\n"
            f"Keywords: {kw_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "keywords": self.keywords,
        }

    def __repr__(self) -> str:
        return f"<FAQItem id='{self.id}' category='{self.category}' question='{self.question[:40]}...'>"


class KnowledgeBase:
    """Manages the university helpdesk FAQ document collection."""

    def __init__(self, faq_file_path: Optional[str] = None):
        if faq_file_path is None:
            # Default to data/university_faq.json relative to repository root
            root_dir = Path(__file__).resolve().parent.parent
            faq_file_path = str(root_dir / "data" / "university_faq.json")

        self.file_path = Path(faq_file_path)
        self.items: List[FAQItem] = []
        self.load()

    def load(self) -> List[FAQItem]:
        """Loads and parses the FAQ items from JSON file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"FAQ file not found at: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.items = [
            FAQItem(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                answer=item["answer"],
                keywords=item.get("keywords", [])
            )
            for item in raw_data
        ]
        return self.items

    def get_all(self) -> List[FAQItem]:
        """Returns all FAQ items."""
        return self.items

    def get_by_id(self, item_id: str) -> Optional[FAQItem]:
        """Fetches an item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def get_categories(self) -> List[str]:
        """Returns a sorted list of unique categories."""
        return sorted(list({item.category for item in self.items}))

    def get_chunks_for_indexing(self) -> List[Dict[str, Any]]:
        """Returns chunks with text and metadata suitable for vector and keyword indexing."""
        chunks = []
        for item in self.items:
            chunks.append({
                "id": item.id,
                "text": item.searchable_text,
                "category": item.category,
                "question": item.question,
                "answer": item.answer,
                "keywords": item.keywords
            })
        return chunks

    def __len__(self) -> int:
        return len(self.items)
