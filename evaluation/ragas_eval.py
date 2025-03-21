import argparse
import json
import os
import requests
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
    answer_similarity,
)

# --- Config ---
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
MAX_TEXTS = 1
LLM_EVALUATOR = "gpt-4o-mini"
EMBEDDINGS_EVALUATOR = "text-embedding-3-small"
EVAL_DIR = "evaluation"
JSON_LOG_FILE = os.path.join(EVAL_DIR, "experiment_log.json")
CSV_LOG_FILE = os.path.join(EVAL_DIR, "experiment_history.csv")


# --- IO Utils ---
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# --- Indexing ---
def index_documents(data, max_texts=None):
    texts = []
    for document in data:
        contexts = [p["context"] for p in document["paragraphs"]]
        texts.extend(contexts)

    if max_texts is not None:
        texts = texts[:max_texts]

    upload_payload = {"texts": texts}
    print("Uploading documents...\n")
    resp = requests.post(f"{BASE_URL}/upload", json=upload_payload)
    print("Status code /upload:", resp.status_code)
    print("Response /upload:", resp.json())


# --- RAG App Response ---
def generate_response():
    return {
        "generated_text": "Pretend I generated something",
        "contexts": ["Fake context 1", "Fake context 2"]
    }


# --- Ragas Dataset Generation ---
def generate_ragas_dataset(data, max_texts=None):
    dataset = []
    texts_added = 0

    for document in data:
        for paragraph in document["paragraphs"]:
            if max_texts is not None and texts_added >= max_texts:
                break

            texts_added += 1
            for qa in paragraph["qas"]:
                rag_answer = generate_response()
                dataset.append({
                    "user_input": qa["question"],
                    "retrieved_contexts": rag_answer["contexts"],
                    "response": rag_answer["generated_text"],
                    "reference": qa["answers"][0]["text"]
                })

        if max_texts is not None and texts_added >= max_texts:
            break

    return EvaluationDataset.from_list(dataset)


# --- Evaluation Helpers ---
def get_evaluation_result(evaluation_dataset):
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=LLM_EVALUATOR))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDINGS_EVALUATOR))

    return evaluate(
        dataset=evaluation_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
            answer_similarity,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )


def prepare_experiment_log(result, experiment_notes):
    df = result.to_pandas()
    avg_scores = df.select_dtypes(include="number").mean().to_dict()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "experiment_notes": experiment_notes,
        "timestamp": timestamp,
        "average_scores": avg_scores,
        "scores": df.to_dict(orient="records")
    }


def save_experiment_log_to_json(log_data):
    os.makedirs(EVAL_DIR, exist_ok=True)
    save_json(log_data, JSON_LOG_FILE)
    print(f"Experiment logged in {JSON_LOG_FILE}")


def append_experiment_summary_to_csv(log_data):
    average_metrics = log_data["average_scores"]
    numeric_scores = [v for v in average_metrics.values() if isinstance(v, (int, float))]
    overall_average_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None

    summary_row = {
        "timestamp": log_data["timestamp"],
        "experiment_notes": log_data["experiment_notes"],
        "average_score": round(overall_average_score, 4),
        **{k: round(v, 4) for k, v in average_metrics.items()}
    }

    df = pd.DataFrame([summary_row])
    df.index.name = "experiment_id"

    if os.path.exists(CSV_LOG_FILE):
        df.to_csv(CSV_LOG_FILE, mode="a", header=False)
    else:
        df.to_csv(CSV_LOG_FILE, mode="w", index=True)

    print("Experiment appended to", CSV_LOG_FILE)


# --- Main Evaluation Entry Point ---
def evaluate_rag_app(evaluation_dataset, experiment_notes=""):
    result = get_evaluation_result(evaluation_dataset)
    log_data = prepare_experiment_log(result, experiment_notes)
    save_experiment_log_to_json(log_data)
    append_experiment_summary_to_csv(log_data)


# --- Entrypoint ---
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG app and log experiment results.")
    parser.add_argument(
        "--notes",
        required=True,
        help="Required notes describing the experiment (e.g., model configs, changes, observations)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    experiment_notes = args.notes

    data_path = os.path.join(EVAL_DIR, "data", "test.json")
    data = read_json(data_path)["data"]
    index_documents(data, max_texts=MAX_TEXTS)
    evaluation_dataset = generate_ragas_dataset(data, max_texts=MAX_TEXTS)
    evaluate_rag_app(evaluation_dataset, experiment_notes=experiment_notes)


if __name__ == "__main__":
    main()
