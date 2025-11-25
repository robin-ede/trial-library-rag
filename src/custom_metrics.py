"""
Custom evaluation metrics for RAG system.

These metrics are domain-specific and complement the standard Ragas metrics.
"""
from typing import List
from langchain_core.documents import Document


def citation_accuracy(answer: str, expected_source: str) -> float:
    """
    Check if the answer cites the expected source document.

    Args:
        answer: The generated answer text
        expected_source: The filename that should be cited (e.g., "nscl.pdf")

    Returns:
        1.0 if expected source is cited, 0.0 otherwise
    """
    if not answer or not expected_source:
        return 0.0

    # Normalize both for comparison
    answer_lower = answer.lower()
    source_lower = expected_source.lower()

    # Check for exact match
    if source_lower in answer_lower:
        return 1.0

    # Check for basename match (e.g., "nscl" in "nscl.pdf")
    source_basename = source_lower.replace(".pdf", "").replace(".txt", "")
    if source_basename in answer_lower:
        return 1.0

    return 0.0


def retrieval_recall(retrieved_docs: List[Document], expected_source: str) -> float:
    """
    Check if the expected source document appears in retrieved results.

    This measures whether the retrieval system found the right document,
    regardless of whether the LLM used it correctly.

    Args:
        retrieved_docs: List of retrieved Document objects
        expected_source: The filename that should be retrieved

    Returns:
        1.0 if expected source is in retrieved docs, 0.0 otherwise
    """
    if not retrieved_docs or not expected_source:
        return 0.0

    expected_lower = expected_source.lower()

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        if expected_lower in source.lower():
            return 1.0

    return 0.0


def has_appropriate_refusal(
    answer: str,
    context: str,
    question: str,
    llm=None
) -> dict:
    """
    Check if the system appropriately refuses to answer when context is insufficient.

    This is important for healthcare applications where incorrect information is dangerous.

    Args:
        answer: The generated answer
        context: The context provided to the LLM
        question: The original question
        llm: Optional LLM to use for judging appropriateness (if None, uses heuristics)

    Returns:
        dict with keys: 'refused' (bool), 'appropriate' (bool), 'score' (float)
    """
    refusal_phrases = [
        "do not know",
        "cannot answer",
        "ask a clinician",
        "insufficient information",
        "unable to answer",
        "not enough context",
    ]

    answer_lower = answer.lower()
    refused = any(phrase in answer_lower for phrase in refusal_phrases)

    # Heuristic: If context is very short, refusal is likely appropriate
    context_insufficient = len(context.strip()) < 100

    if not refused:
        # Didn't refuse - this is appropriate if context is sufficient
        appropriate = not context_insufficient
        score = 1.0 if appropriate else 0.5
    else:
        # Refused - appropriate if context is insufficient, inappropriate if sufficient
        appropriate = context_insufficient
        score = 1.0 if appropriate else 0.0

    return {
        "refused": refused,
        "appropriate": appropriate,
        "score": score,
    }


def answer_contains_ground_truth(answer: str, ground_truth: str) -> float:
    """
    Simple check if the answer contains the ground truth information.

    Args:
        answer: The generated answer
        ground_truth: The expected answer content

    Returns:
        1.0 if ground truth appears in answer, 0.0 otherwise
    """
    if not answer or not ground_truth:
        return 0.0

    # Normalize both
    answer_lower = answer.lower()
    truth_lower = ground_truth.lower()

    # Check for substring match
    if truth_lower in answer_lower:
        return 1.0

    # Check for key phrase match (split on common delimiters)
    truth_phrases = [p.strip() for p in truth_lower.split(",") if len(p.strip()) > 3]
    matches = sum(1 for phrase in truth_phrases if phrase in answer_lower)

    if matches > 0:
        return matches / len(truth_phrases) if truth_phrases else 0.0

    return 0.0
