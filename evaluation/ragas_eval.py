import argparse
import json
import logging
import os
import requests
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult
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
from ragas.run_config import RunConfig


# --- Config ---
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
MAX_TEXTS = 20
LLM_EVALUATOR = "gpt-4o-mini"
EMBEDDINGS_EVALUATOR = "text-embedding-3-small"
EVAL_DIR = "evaluation"
JSON_LOG_FILE = os.path.join(EVAL_DIR, "experiment_log.json")
CSV_LOG_FILE = os.path.join(EVAL_DIR, "experiment_history.csv")

# Global speed performance trackers
TOTAL_RESPONSE_TIME = 0.0
TOTAL_RESPONSES = 0

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- IO Utils ---
def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# --- Indexing ---
def index_documents(data: list, max_texts: int=None):
    """
    Indexes documents by extracting their contexts and sending them to the RAG system.

    Args:
        data (list): The dataset containing documents with paragraphs.
        max_texts (int, optional): Maximum number of texts (paragraphs) to index. If None, indexes all available texts.
    """
    texts = []
    for document in data:
        contexts = [p["context"] for p in document["paragraphs"]]
        texts.extend(contexts)

    if max_texts is not None:
        texts = texts[:max_texts]

    upload_payload = {"texts": texts}
    logging.info("Uploading documents...")
    resp = requests.post(f"{BASE_URL}/upload", json=upload_payload)
    logging.info(f"Status code /upload: {resp.status_code}")
    logging.info(f"Response /upload: {resp.json()}")


# --- RAG App Response ---
def generate_response(question: str):
    """
    Sends a request to the RAG system's /generate endpoint to obtain a response.
    Updates global counters for response speed calculation.

    Args:
        question (str): The user query to send to the RAG model.

    Returns:
        dict: A dictionary containing the generated response and retrieved contexts.
    """
    global TOTAL_RESPONSE_TIME, TOTAL_RESPONSES

    logging.info(f"Question: {question}")
    payload = {"new_message": {"role": "user", "content": question}}

    start_time = time.perf_counter()

    try:
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        response.raise_for_status()
        result = response.json()
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling /generate: {e}")
        result = {"generated_text": "Error: Unable to generate response.", "contexts": []}
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    TOTAL_RESPONSE_TIME += elapsed
    TOTAL_RESPONSES += 1

    return result


# --- Ragas Dataset Generation ---
def generate_ragas_dataset(data: list, max_texts=None):
    """
    Generates an evaluation dataset for RAGAS by extracting Q&A pairs from documents.

    Args:
        data (list): The dataset containing documents with paragraphs and questions.
        max_texts (int, optional): Maximum number of texts (paragraphs) to include in the dataset. If None, uses all.

    Returns:
        EvaluationDataset: A RAGAS evaluation dataset object.
    """
    dataset = []
    texts_added = 0

    for document in data:
        for paragraph in document["paragraphs"]:
            if max_texts is not None and texts_added >= max_texts:
                break

            texts_added += 1
            logging.info(f"Processing text: {texts_added}/{MAX_TEXTS}")
            for qa in paragraph["qas"]:
                rag_answer = generate_response(qa["question"])
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
def get_evaluation_result(evaluation_dataset: EvaluationDataset):
    """
    Computes evaluation metrics for the given dataset using RAGAS.

    Args:
        evaluation_dataset (EvaluationDataset): The dataset containing queries, retrieved contexts, responses, and references.

    Returns:
        EvaluationResult: The result object containing computed metrics.
    """
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=LLM_EVALUATOR))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDINGS_EVALUATOR))

    return evaluate(
        dataset=evaluation_dataset,
        raise_exceptions=True,
        run_config=RunConfig(
            timeout=60,
            max_retries=10,
            max_wait = 180,
            max_workers= 16,
        ),
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
            answer_similarity,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        batch_size=32
    )


def prepare_experiment_log(result: EvaluationResult, experiment_notes: str):
    """
    Prepares structured logging data for the experiment.

    Args:
        result (EvaluationResult): The evaluation result containing metric scores.
        experiment_notes (str): A description of the experiment.

    Returns:
        dict: A dictionary with experiment metadata, scores, and raw results.
    """
    df = result.to_pandas()
    avg_scores = df.select_dtypes(include="number").mean().to_dict()

    numeric_scores = [v for v in avg_scores.values() if isinstance(v, (int, float))]
    overall_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    responses_per_second = TOTAL_RESPONSES / TOTAL_RESPONSE_TIME if TOTAL_RESPONSE_TIME > 0 else 0.0

    return {
        "experiment_notes": experiment_notes,
        "timestamp": timestamp,
        "overall_score": overall_score,
        "responses_per_second": round(responses_per_second, 4),
        "average_scores": avg_scores,
        "scores": df.to_dict(orient="records")
    }


def save_experiment_log_to_json(log_data: dict):
    os.makedirs(EVAL_DIR, exist_ok=True)
    save_json(log_data, JSON_LOG_FILE)
    logging.info(f"Experiment logged in {JSON_LOG_FILE}")


def append_experiment_summary_to_csv(log_data: dict):
    """
    Appends a summary of the experiment (including average scores) to a CSV file. Auto-increments the experiment_id. 

    Args:
        log_data (dict): The structured log data containing average scores and metadata.
    """
    average_metrics = log_data["average_scores"]

    if os.path.exists(CSV_LOG_FILE):
        existing_df = pd.read_csv(CSV_LOG_FILE)
        if "experiment_id" in existing_df.columns and not existing_df.empty:
            next_experiment_id = existing_df["experiment_id"].max() + 1
        else:
            next_experiment_id = 0
    else:
        next_experiment_id = 0 

    summary_row = {
        "experiment_id": next_experiment_id,
        "timestamp": log_data["timestamp"],
        "experiment_notes": log_data["experiment_notes"],
        "overall_score": round(log_data["overall_score"], 4),
        "responses_per_second": round(log_data["responses_per_second"], 4),
        **{k: round(v, 4) for k, v in average_metrics.items()}
    }

    df = pd.DataFrame([summary_row])
    df.index.name = "experiment_id"

    if os.path.exists(CSV_LOG_FILE):
        df.to_csv(CSV_LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_LOG_FILE, mode="w", index=False)

    logging.info(f"Experiment appended to {CSV_LOG_FILE}")


# --- Main Evaluation Entry Point ---
def evaluate_rag_app(evaluation_dataset: EvaluationDataset, experiment_notes: str=""):
    """
    Orchestrates the full evaluation process: computing metrics, logging results, and saving summaries.

    Args:
        evaluation_dataset (EvaluationDataset): The dataset to evaluate.
        experiment_notes (str): Notes describing the experiment.
    """
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
