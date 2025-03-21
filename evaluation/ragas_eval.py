import json
import os
import requests

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
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
MAX_TEXTS = 100


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)    


def index_documents(data, max_texts=None):
    """
    Load documents from the evaluation dataset then send a request to the /upload endpoint to index and persist the evaluation documents.

    Args:
        data (dict): Loaded SQAC corpus test.json file.
        max_texts (int, optional): The maximum number of texts to index. If None, indexes the full evaluation dataset.
    """
    texts = []
    for document in data:
        contexts = [paragraph["context"] for paragraph in document["paragraphs"]]
        texts.extend(contexts)

    if max_texts is not None:
        texts = texts[:max_texts]

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


def generate_ragas_dataset(data, max_texts=None):
    """
    Prepare the evaluation dataset by generating responses and structuring data for evaluation.

    Args:
        data (dict): Loaded SQAC corpus test.json file.
        max_texts (int, optional): The maximum number of texts to use for evaluation. If None, uses the full evaluation dataset.
    
    Returns:
        ragas.EvaluationDataset: Prepared dataset for RAG evaluation.
    """
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



def evaluate_rag_app(evaluation_dataset):
    """
    Compute evaluation metrics for the provided evaluation dataset using OpenAI's gpt-4o-mini as the llm evaluator and text-embedding-3-small as the embeddings evaluator.

    Args:
        evaluation_dataset (ragas.EvaluationDataset): The dataset containing user queries, retrieved contexts, generated responses, and reference answers.
    """
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    result = evaluate(
        dataset=evaluation_dataset, 
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_utilization,
            context_recall,    
            answer_similarity,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
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
    data = read_json(data_path)["data"]
    index_documents(data, max_texts=MAX_TEXTS)
    evaluation_dataset = generate_ragas_dataset(data, max_texts=MAX_TEXTS)
    evaluate_rag_app(evaluation_dataset)


if __name__ == "__main__":
    main()