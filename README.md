## RAG API with Docker

This project provides a Retrieval-Augmented Generation (RAG) API using FastAPI, LlamaIndex, and Hugging Face models. The system consists of three containers:

1. **Text Embeddings Inference (TEI)**: Responsible for computing text embeddings.
2. **Text Generation Inference (TGI)**: Generates responses based on retrieved context.
3. **RAG API**: Manages indexing and query processing.

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Setup and Running the Containers

#### 0. Create a .env file with the following fields:
```
BASE_URL=http://localhost:8000

OPENAI_API_KEY=<your api key here in case you want to run the `ragas_eval.py`> (optional)
HF_TOKEN=<your hugging face token in case you want to use TGI models with restricted access> (optional)

TEI_MODEL=sentence-transformers/all-MiniLM-L12-v2 (change this value to change the tei model used)
TGI_MODEL=Qwen/Qwen2.5-0.5B-Instruct (change this value to change the tgi model used)
```

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

You can run the `tests/test_rag.py` file to test how this works running on python.

### Summary

- **Build the images**: `docker compose build`
- **Run the containers**: `docker compose up -d`
- **Stop the containers**: `docker compose down`
- **Test the API**: Use `curl` commands or `tests/test_rag.py`


### HOW TO WORK WITH THIS MOVING FORWARD

1. When you open the repository, run the following in your terminal, to build the docker containers:

```sh
docker compose build
```

```sh
docker compose up -d
```

2. Work on the project

3. When you want to leave/close the project, run the following in the terminal to close the containers:
```sh
docker compose down
```