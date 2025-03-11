## RAG API with Docker

This project provides a Retrieval-Augmented Generation (RAG) API using FastAPI, LlamaIndex, and Hugging Face models. The system consists of three containers:

1. **Text Embeddings Inference (TEI)**: Responsible for computing text embeddings.
2. **Text Generation Inference (TGI)**: Generates responses based on retrieved context.
3. **RAG API**: Manages indexing and query processing.

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Setup and Running the Containers

#### 1. Build the Containers

```sh
docker compose build
```

#### 2. Start the Containers

```sh
docker compose up -d
```

This will start the following services:

- `tei` on port `8080`
- `tgi` on port `8081`
- `rag` (the API) on port `8000`

#### 3. Stop the Containers

```sh
docker compose down
```

### API Endpoints

Once the RAG API is running, you can interact with it via HTTP requests.

#### **1. Upload Text Documents**

To create the vector database, send a list of texts to be indexed.

```sh
curl -X POST "http://localhost:8000/upload" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["The capital of France is Paris.", "Python was created by Guido van Rossum."]}'
```

#### **2. Generate Responses**

Once documents are indexed, ask questions based on the stored knowledge.

```sh
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"new_message": {"role": "user", "content": "What is the capital of France?"}}'
```

### Testing the API with Python

Save the following script as `test_rag.py` and run it after starting the containers:

```python
import requests

def main():
    base_url = "http://localhost:8000"  # Adjust if needed

    # Upload sample texts
    texts = [
        "The capital of France is Paris. France is in Europe.",
        "Don Quixote was written by Miguel de Cervantes in the early 17th century.",
        "Python is a popular programming language created by Guido van Rossum."
    ]
    upload_payload = {"texts": texts}
    print("Uploading documents...\n")
    resp_upload = requests.post(f"{base_url}/upload", json=upload_payload)
    print("Status code /upload:", resp_upload.status_code)
    print("Response /upload:", resp_upload.json())

    # Example questions
    questions = [
        "What is the capital of France?",
        "Who created the Python language?",
        "Who wrote Don Quixote?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        generate_payload = {"new_message": {"role": "user", "content": q}}
        resp_generate = requests.post(f"{base_url}/generate", json=generate_payload)
        print("Status code /generate:", resp_generate.status_code)
        if resp_generate.ok:
            data = resp_generate.json()
            print("Generated response:", data.get("generated_text"))
        else:
            print("Error:", resp_generate.text)

if __name__ == "__main__":
    main()
```

### Summary

- **Build the images**: `docker compose build`
- **Run the containers**: `docker compose up -d`
- **Stop the containers**: `docker compose down`
- **Test the API**: Use `curl` commands or `test_rag.py`
