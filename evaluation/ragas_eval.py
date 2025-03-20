import json
import os
import requests

from dotenv import load_dotenv
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_utilization,
    context_recall,    
    answer_similarity,
)


load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)    


def index_documents(path):
    """
    Load documents from the evaluation dataset then send a request to the /upload endpoint to index and persist the evaluation documents.

    Args:
        path (str): Path to the SQAC corpus test.json file.
    """
    data = read_json(path)["data"]

    texts = []
    for document in data:
        contexts = [paragraph["context"] for paragraph in document["paragraphs"]]
        texts.extend(contexts)
    
    upload_payload = {"texts": texts}
    print("Uploading documents...\n")
    resp_upload = requests.post(f"{BASE_URL}/upload", json=upload_payload)
    print("Status code /upload:", resp_upload.status_code)
    print("Response /upload:", resp_upload.json())


def generate_response():
    """
    Send a request to the /generate endpoint of the RAG app.

    Returns:
        dict: A dictionary containing generated text and retrieved contexts.
    """
    return {
        "generated_text": "Pretend I generated something",
        "contexts": ["Fake context 1", "Fake context 2"]
    }


def generate_ragas_dataset(path):
    """
    Load and prepare the evaluation dataset by generating responses and structuring data for evaluation.

    Args:
        path (str): Path to the SQAC corpus test.json file.

    Returns:
        ragas.EvaluationDataset: Prepared dataset for RAG evaluation.
    """
    data = read_json(path)["data"]

    dataset = []
    
    for text in data:
        for paragraph in text["paragraphs"]:
            for qa in paragraph["qas"]:
                rag_answer = generate_response()
                dataset.append({
                    "user_input": qa["question"],
                    "retrieved_contexts": rag_answer["contexts"],
                    "response": rag_answer["generated_text"],
                    "reference": qa["answers"][0]["text"]
                })
                    
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    return evaluation_dataset


def evaluate_rag_app(evaluation_dataset):
    """
    Compute evaluation metrics for the provided evaluation dataset using OpenAI's GPT-4o-mini as the evaluator.

    Args:
        evaluation_dataset (ragas.EvaluationDataset): The dataset containing user queries, retrieved contexts, generated responses, and reference answers.
    """
    result = evaluate(
        dataset = evaluation_dataset, 
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_utilization,
            context_recall,    
            answer_similarity,
        ],
    )

    df = result.to_pandas()

    save_path = os.path.join("evaluation", "metrics.csv")
    df.to_csv(save_path, index=False)


def main():
    """
    Perform an end-to-end evaluation of the RAG application using the SQAC corpus test set.
    
    Steps:
        1. Index the documents from the evaluation dataset.
        2. Generate the evaluation dataset by obtaining responses from the RAG system.
        3. Compute RAGAS evaluation metrics and save them as a CSV file.
    """
    data_path = os.path.join("evaluation", "data", "test.json")
    index_documents(data_path)
    evaluation_dataset = generate_ragas_dataset(data_path)
    evaluate_rag_app(evaluation_dataset)


if __name__ == "__main__":
    main()