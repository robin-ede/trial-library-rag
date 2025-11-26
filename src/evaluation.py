"""
Evaluation system for RAG application.

Uses curated question-answer pairs with known ground truth for reliable measurement.
This approach is more reliable than synthetic evaluation (randomly generating questions
from chunks), as it ensures questions are actually answerable and ground truth is accurate.
"""
import os
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.retrieval import get_advanced_retriever
from src.tracked_embeddings import TrackedOpenAIEmbeddings
from src.generation import get_rag_chain, format_docs
from src.custom_metrics import (
    citation_accuracy,
    retrieval_recall,
    has_appropriate_refusal,
    answer_contains_ground_truth,
)

load_dotenv()

# Evaluation questions with ground truth
# These are based on the NSCL guideline document and other PDFs in data/
EVAL_QUESTIONS = [
    {
        "question": "What is the recommended duration of maintenance immunotherapy for patients who received front-line immunotherapy?",
        "ground_truth": "2 years if tolerated",
        "expected_source": "nscl.pdf",
        "category": "treatment_duration",
        "difficulty": "easy",
    },
    {
        "question": "Which targeted therapy is preferred as first-line treatment for patients with EGFR exon 19 deletion or L858R mutations?",
        "ground_truth": "osimertinib",
        "expected_source": "nscl.pdf",
        "category": "targeted_therapy",
        "difficulty": "easy",
    },
    {
        "question": "What imaging studies are recommended to rule out metastatic disease in patients with T1-2, N2 disease confirmed before thoracotomy?",
        "ground_truth": "brain MRI with contrast and FDG PET/CT scan",
        "expected_source": "nscl.pdf",
        "category": "diagnostic_imaging",
        "difficulty": "medium",
    },
    {
        "question": "For patients with performance status 2 and nonsquamous NSCLC, what are the recommended single-agent chemotherapy options?",
        "ground_truth": "gemcitabine, pemetrexed, or taxanes",
        "expected_source": "nscl.pdf",
        "category": "chemotherapy",
        "difficulty": "medium",
    },
    # Additional questions from NSCL guidelines
    {
        "question": "Which biomarkers should be included in molecular testing for patients with metastatic nonsquamous NSCLC?",
        "ground_truth": "EGFR mutation, ALK, KRAS, ROS1, BRAF, NTRK1/2/3, MET exon 14 skipping, RET, ERBB2 (HER2), NRG1, HER2 (IHC), and HGFR(MET) IHC",
        "expected_source": "nscl.pdf",
        "category": "biomarker_testing",
        "difficulty": "hard",
    },
    {
        "question": "What are the criteria for administering bevacizumab to patients with NSCLC?",
        "ground_truth": "Nonsquamous NSCLC and no recent history of hemoptysis",
        "expected_source": "nscl.pdf",
        "category": "contraindications",
        "difficulty": "medium",
    },
    # Questions from FDA Diversity Action Plan guidance
    {
        "question": "By which demographic characteristics must clinical study enrollment goals be disaggregated in a Diversity Action Plan?",
        "ground_truth": "Race, ethnicity, sex, and age group",
        "expected_source": "fda_guidance.pdf",
        "category": "regulatory_requirements",
        "difficulty": "easy",
    },
    {
        "question": "What are the three statutory criteria under which the FDA may grant a waiver for a Diversity Action Plan?",
        "ground_truth": "1. A waiver is necessary based on the prevalence or incidence of the disease or condition in the U.S.; 2. Conducting a clinical investigation in accordance with a Diversity Action Plan would otherwise be impracticable; 3. A waiver is necessary to protect public health during a public health emergency",
        "expected_source": "fda_guidance.pdf",
        "category": "waivers",
        "difficulty": "hard",
    },
    # Questions from diversity study
    {
        "question": "What five prominent themes regarding minority recruitment emerged from the qualitative interviews with cancer center stakeholders?",
        "ground_truth": "1. Interactions with potential minority participants were perceived to be challenging; 2. Potential minority participants were not perceived to be ideal study candidates; 3. A combination of clinic-level barriers and negative perceptions led to withholding trial opportunities; 4. Tailored recruitment practices often focused on addressing research misconceptions to build trust; 5. Some respondents viewed race as irrelevant when screening/recruiting",
        "expected_source": "diversity_study.pdf",
        "category": "qualitative_analysis",
        "difficulty": "hard",
    },
    {
        "question": "According to the study, what specific negative stereotype did referring clinicians often associate with minority patients regarding clinical trials?",
        "ground_truth": "Lower potential for adherence or compliance to study protocols",
        "expected_source": "diversity_study.pdf",
        "category": "stereotyping",
        "difficulty": "medium",
    },
]


def filter_placeholders(eval_set: List[Dict]) -> List[Dict]:
    """Remove placeholder questions from evaluation set."""
    return [q for q in eval_set if not q["question"].startswith("PLACEHOLDER")]


def run_evaluation(use_placeholders: bool = False):
    """
    Run evaluation using curated question-answer pairs.

    Args:
        use_placeholders: If False, skips placeholder questions (default: False)
    """
    print("=" * 80)
    print("RAG SYSTEM EVALUATION")
    print("=" * 80)

    # Filter out placeholders if requested
    eval_set = EVAL_QUESTIONS if use_placeholders else filter_placeholders(EVAL_QUESTIONS)

    if not eval_set:
        print("\nNo questions to evaluate! Please fill in the placeholder questions first.")
        return

    print(f"\nEvaluating {len(eval_set)} curated questions...")

    # Initialize components
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    embeddings = TrackedOpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    if not vectorstore._collection.count():
        print("\n❌ Vector store is empty. Please run ingestion.py first.")
        return

    retriever = get_advanced_retriever(k=5)
    rag_chain = get_rag_chain()

    # Collect results
    questions = []
    ground_truths = []
    retrieved_contexts = []
    answers = []
    categories = []
    difficulties = []

    # Custom metric results
    citation_scores = []
    retrieval_recall_scores = []
    refusal_scores = []
    ground_truth_match_scores = []

    print("\nProcessing questions...\n")

    for i, item in enumerate(eval_set, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_source = item["expected_source"]
        category = item["category"]
        difficulty = item["difficulty"]

        print(f"[{i}/{len(eval_set)}] {category} ({difficulty})")
        print(f"Q: {question[:80]}...")

        # Get retrieved docs first
        try:
            docs = retriever.invoke(question)
            ctxs = [d.page_content for d in docs]
        except Exception as e:
            print(f"  ❌ Error retrieving docs: {e}")
            docs = []
            ctxs = []

        # Run RAG with retrieved context
        try:
            context = format_docs(docs) if docs else ""
            result = rag_chain.invoke({"context": context, "question": question})
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            print(f"  ❌ Error generating answer: {e}")
            answer = "Error during generation"

        # Store for Ragas
        questions.append(question)
        ground_truths.append(ground_truth)
        retrieved_contexts.append(ctxs)
        answers.append(answer)
        categories.append(category)
        difficulties.append(difficulty)

        # Calculate custom metrics
        citation_score = citation_accuracy(answer, expected_source)
        recall_score = retrieval_recall(docs, expected_source)
        refusal_result = has_appropriate_refusal(answer, "\n".join(ctxs), question)
        gt_match_score = answer_contains_ground_truth(answer, ground_truth)

        citation_scores.append(citation_score)
        retrieval_recall_scores.append(recall_score)
        refusal_scores.append(refusal_result["score"])
        ground_truth_match_scores.append(gt_match_score)

        # Print preview
        print(f"  A: {answer[:100]}...")
        print(f"  📊 Citation: {citation_score:.2f} | Recall: {recall_score:.2f} | GT Match: {gt_match_score:.2f}")
        print()

    # Create dataset for Ragas
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths,
        }
    )

    print("=" * 80)
    print("Running Ragas evaluation...")
    print("=" * 80)

    try:
        ragas_results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
        )
        print("\n✅ Ragas evaluation complete!")
    except Exception as e:
        print(f"\n❌ Ragas evaluation failed: {e}")
        ragas_results = None

    # Combine results
    results_df = pd.DataFrame(
        {
            "question": questions,
            "category": categories,
            "difficulty": difficulties,
            "answer": answers,
            "ground_truth": ground_truths,
            "citation_accuracy": citation_scores,
            "retrieval_recall": retrieval_recall_scores,
            "refusal_appropriate": refusal_scores,
            "ground_truth_match": ground_truth_match_scores,
        }
    )

    # Add Ragas metrics if available
    if ragas_results is not None:
        ragas_df = ragas_results.to_pandas()
        results_df["faithfulness"] = ragas_df["faithfulness"]
        results_df["answer_relevancy"] = ragas_df["answer_relevancy"]
        results_df["context_precision"] = ragas_df["context_precision"]

    # Save results
    output_file = "evaluation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print("\n📊 CUSTOM METRICS (Averages):")
    print(f"  Citation Accuracy:      {results_df['citation_accuracy'].mean():.3f}")
    print(f"  Retrieval Recall:       {results_df['retrieval_recall'].mean():.3f}")
    print(f"  Refusal Appropriate:    {results_df['refusal_appropriate'].mean():.3f}")
    print(f"  Ground Truth Match:     {results_df['ground_truth_match'].mean():.3f}")

    if ragas_results is not None:
        print("\n📊 RAGAS METRICS (Averages):")
        print(f"  Faithfulness:           {results_df['faithfulness'].mean():.3f}")
        print(f"  Answer Relevancy:       {results_df['answer_relevancy'].mean():.3f}")
        print(f"  Context Precision:      {results_df['context_precision'].mean():.3f}")

    print("\n📊 BY CATEGORY:")
    category_summary = results_df.groupby("category").agg(
        {
            "citation_accuracy": "mean",
            "retrieval_recall": "mean",
            "ground_truth_match": "mean",
        }
    )
    print(category_summary.round(3))

    print("\n📊 BY DIFFICULTY:")
    difficulty_summary = results_df.groupby("difficulty").agg(
        {
            "citation_accuracy": "mean",
            "retrieval_recall": "mean",
            "ground_truth_match": "mean",
        }
    )
    print(difficulty_summary.round(3))

    print("\n" + "=" * 80)
    print(f"✅ Evaluation complete! {len(eval_set)} questions processed.")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    # By default, skip placeholder questions
    # Set use_placeholders=True to include them (they will likely fail)
    results = run_evaluation(use_placeholders=False)
