"""
Simple IT ticket search helper using TF-IDF cosine similarity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

TICKET_FILE = Path(__file__).with_name("tickets.csv")
TOP_K = 3


def load_tickets(csv_path: Path) -> pd.DataFrame:
    """Load tickets and create combined searchable text."""
    df = pd.read_csv(csv_path)
    if not {"ticket_id", "title", "description", "fix"}.issubset(df.columns):
        raise ValueError("CSV missing required columns.")
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["fix"] = df["fix"].fillna("No fix captured.")
    df["search_text"] = df["title"] + " " + df["description"]
    return df


def build_model(text_series: pd.Series) -> tuple[TfidfVectorizer, any]:
    """Fit TF-IDF model on the provided text and return vectorizer + matrix."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(text_series)
    return vectorizer, tfidf_matrix


def run_search_loop(df: pd.DataFrame, vectorizer: TfidfVectorizer, tfidf_matrix) -> None:
    """Interactive CLI loop for searching tickets."""
    banner = "=" * 64
    print(banner)
    print(" IT Support Ticket Search ")
    print(" Type a short issue description to find similar tickets.")
    print(" Type 'quit' to exit.")
    print(banner)

    while True:
        query = input("Enter your issue (or type 'quit'): ").strip()
        if not query:
            print("Please enter a description or type 'quit'.")
            continue

        if query.lower() == "quit":
            print("Goodbye!")
            break

        query_vector = vectorizer.transform([query])
        similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:TOP_K]

        if similarities[top_indices[0]] <= 0:
            print("No similar tickets found. Try different wording.")
            continue

        print("\nTop matches:\n")
        for rank, idx in enumerate(top_indices, start=1):
            score = similarities[idx]
            if score <= 0:
                continue
            row = df.iloc[idx]
            print(f"[{rank}] Ticket: {row.ticket_id}  |  Similarity: {score:.3f}")
            print(f"    Title : {row.title}")
            print(f"    Fix   : {row.fix}\n")


def main() -> int:
    try:
        df = load_tickets(TICKET_FILE)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Failed to load tickets: {exc}")
        return 1

    vectorizer, tfidf_matrix = build_model(df["search_text"])
    run_search_loop(df, vectorizer, tfidf_matrix)
    return 0


if __name__ == "__main__":
    sys.exit(main())

