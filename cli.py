"""
CampusAssist CLI Interface
Interactive terminal chat loop and automated pipeline verification.
"""

import sys
import argparse
from dotenv import load_dotenv

# Ensure utf-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from src.rag_pipeline import RAGPipeline


def run_interactive_cli(pipeline: RAGPipeline):
    print("=" * 65)
    print("[CampusAssist] University Helpdesk Support Chatbot (CLI)")
    print("=" * 65)
    print("Type your questions below. Type 'exit', 'quit', or 'q' to stop.")
    print("Type 'reset' to clear conversation memory.")
    print("-" * 65)

    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! Have a great day on campus!")
                break
            if query.lower() == "reset":
                pipeline.reset_conversation()
                print("\n[Conversational memory cleared.]")
                continue

            result = pipeline.process_query(query)

            if result["is_followup"]:
                print(f"\n[Contextual Reformulation: '{result['search_query']}']")

            print(f"\nCampusAssist:\n{result['response']}")

            if result["is_relevant"] and result["sources"]:
                top = result["sources"][0]
                print(f"\n[Source: {top['id']} ({top['category']}) | Confidence: {result['confidence_score']:.2f}]")
            elif not result["is_relevant"]:
                print(f"\n[Query flagged as out-of-scope / missing from KB | Score: {result['confidence_score']:.2f}]")

        except KeyboardInterrupt:
            print("\nSession ended.")
            break


def run_automated_test(pipeline: RAGPipeline):
    """Executes a series of automated test turns showing single-turn, follow-up, and out-of-scope queries."""
    test_queries = [
        ("Turn 1 (Direct Query)", "How do I connect to the university Wi-Fi network?"),
        ("Turn 2 (Follow-up / Pronoun)", "What about guests who do not have student credentials?"),
        ("Turn 3 (New Topic)", "How do I get a replacement student ID card?"),
        ("Turn 4 (Follow-up / Ellipsis)", "How much does that cost and where do I pick it up?"),
        ("Turn 5 (Out-of-Domain)", "What is the recipe for baking chocolate chip cookies?")
    ]

    print("\n" + "=" * 70)
    print("[TEST] Running Automated RAG Pipeline Verification Flow")
    print("=" * 70)

    for turn_label, q in test_queries:
        print(f"\n>>> {turn_label}: \"{q}\"")
        res = pipeline.process_query(q)
        print(f"    Search Query Used: \"{res['search_query']}\"")
        print(f"    Is Follow-up: {res['is_followup']}")
        print(f"    Is Relevant: {res['is_relevant']} (Score: {res['confidence_score']:.4f})")
        print(f"    Response:\n{res['response']}")
        print("-" * 70)

    print("\n[PASS] Automated verification completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="CampusAssist University Helpdesk CLI")
    parser.add_argument("--provider", default="auto", choices=["auto", "groq", "local", "openai", "gemini"], help="Generator provider")
    parser.add_argument("--test", action="store_true", help="Run automated test sequence")
    parser.add_argument("--threshold", type=float, default=0.45, help="Relevance cutoff threshold")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        generator_provider=args.provider,
        relevance_threshold=args.threshold
    )

    if args.test:
        run_automated_test(pipeline)
    else:
        run_interactive_cli(pipeline)


if __name__ == "__main__":
    main()
