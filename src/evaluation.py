import random
import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.retrieval import get_retriever
from src.generation import get_rag_chain

load_dotenv()

NUM_QUESTIONS = 5

def sample_chunks(vectorstore, n=NUM_QUESTIONS):
    # Chroma API: get all documents
    docs = vectorstore._collection.get(include=["documents", "metadatas"])
    all_docs = list(
        zip(docs["documents"], docs["metadatas"])
    )
    if not all_docs:
        return []
    return random.sample(all_docs, min(n, len(all_docs)))

def generate_question_from_chunk(llm, text: str) -> str:
    prompt = f"""
You are generating a question for RAG evaluation.

Given this text (from oncology guidelines or a clinical trial document):

\"\"\"{text}\"\"\"

Generate ONE specific, answerable question that can be fully answered using ONLY the given text.
The question should be factual and not vague.
Return only the question.
"""
    resp = llm.invoke(prompt)
    return resp.content.strip()

def main():
    # LLM via OpenRouter
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Load vectorstore & retriever
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    
    # Check if vectorstore has data
    if not vectorstore._collection.count():
        print("Vector store is empty. Please run ingestion.py first.")
        return

    retriever = get_retriever(k=5)
    rag_chain = get_rag_chain(retriever)

    sampled = sample_chunks(vectorstore, NUM_QUESTIONS)

    questions = []
    ground_truths = []
    retrieved_contexts = []
    answers = []

    print(f"Generating {len(sampled)} evaluation questions...")

    for text, meta in sampled:
        q = generate_question_from_chunk(llm, text)
        questions.append(q)
        ground_truths.append(text)

        # Run RAG
        # Note: The chain expects a dict with "question" key
        result = rag_chain.invoke({"question": q})
        answer = result.content if hasattr(result, "content") else str(result)
        answers.append(answer)

        # Get retrieved docs to pass to Ragas as contexts
        docs = retriever.invoke(q)
        ctxs = [d.page_content for d in docs]
        retrieved_contexts.append(ctxs)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths,
        }
    )

    print("Running Ragas evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    print(results)
    results.to_pandas().to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
