import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.core.node_parser.text import SentenceWindowNodeParser

try:
    from llama_index.core import Document
except ImportError:
    from llama_index.readers.schema.base import Document

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

# Initialize FastAPI application
app = FastAPI()

# ==========================
# CONFIGURATION: TEXT EMBEDDINGS INFERENCE (TEI)
# ==========================
tei_model_name = os.getenv("TEI_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Using embedding model: {tei_model_name}")

# Configure embedding settings
Settings.llm = None  # No language model is set directly in settings
Settings.embed_model = TextEmbeddingsInference(
    model_name=tei_model_name,
    base_url="http://tei:80",  # Endpoint for the embedding service
    embed_batch_size=32  # Defines batch size for embedding requests
)

# ==========================
# CONFIGURATION: TEXT GENERATION INFERENCE (TGI)
# ==========================
print("Configuring TGI client for text generation...")

# Initialize the text generation client
generator = InferenceClient("http://tgi:80")

tgi_model_name = os.getenv("TGI_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
print(f"Using generative model (TGI): {tgi_model_name}")

print("Loading tokenizer...")
# Load the tokenizer for processing input text
tokenizer = AutoTokenizer.from_pretrained(tgi_model_name)

# Global variable to store the vector index
index = None

class UploadRequest(BaseModel):
    """
    Request model for document uploads.
    Attributes:
        texts (list[str]): A list of strings to be indexed into the vector database.
    """
    texts: list[str]

@app.post("/upload")
async def upload_documents(req: UploadRequest):
    """
    Endpoint to create a vector database from a list of input texts.
    Steps:
    1. Each string in `req.texts` is converted into a `Document`.
    2. A `SentenceWindowNodeParser` is applied to segment text based on sentences.
    3. A `VectorStoreIndex` is created and persisted to disk for future retrieval.

    Returns:
        JSON response indicating success and the number of nodes created.
    """
    try:
        # Convert texts into Document objects
        documents = [Document(text=text) for text in req.texts]

        if not documents:
            raise HTTPException(status_code=400, detail="No texts were received for indexing.")

        sentence_window = SentenceWindowNodeParser()
        nodes = sentence_window.build_window_nodes_from_documents(documents)

        # Create a vector index
        global index
        index = VectorStoreIndex(nodes)
        
        # Persist the index to disk
        index.storage_context.persist(persist_dir="./index_storage")
        
        return {"message": "Vector index successfully created", "nodes_count": len(nodes)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while creating index: {e}"
        )

@app.post("/generate")
async def generate_text(request: Request):
    """
    Endpoint to generate responses using Retrieval-Augmented Generation (RAG):
    - If the vector index is not in memory, it is loaded from disk.
    - Relevant nodes matching the query are retrieved.
    - A prompt is constructed using a system message and retrieved context.
    - The prompt is sent to the TGI service for text generation.

    Returns:
        JSON response containing the generated text.
    """
    global index
    try:
        data = await request.json()
        new_message = data.get("new_message", {})
        if "content" not in new_message:
            raise HTTPException(
                status_code=400,
                detail="The attribute 'content' is missing in 'new_message'."
            )
        
        # Load index from storage if not already in memory
        if index is None:
            storage_context = StorageContext.from_defaults(persist_dir="./index_storage")
            index = load_index_from_storage(storage_context)
        
        # Retrieve relevant nodes using similarity search
        query_engine = index.as_query_engine(streaming=False, similarity_top_k=10)
        nodes_retrieved = query_engine.retrieve(new_message["content"])
        
        # Extract text from retrieved nodes
        docs = "".join([f"<doc>\n{node.text}</doc>" for node in nodes_retrieved])
        
        # Construct system prompt for the assistant
        system_prompt = (
            "You are an assistant that responds strictly based on the "
            "provided document information."
        )
        
        # Construct input prompt for generation
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "input", "content": docs},
                new_message
            ],
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Generate response using the TGI service
        answer = generator.text_generation(
            prompt,
            max_new_tokens=128,  # Limit response length
            top_p=0.8,  # Probability threshold for nucleus sampling
            temperature=0.1,  # Control randomness of output
            stop=[tokenizer.eos_token or "<|eot_id|>"],
            do_sample=True,  # Enable sampling for diverse responses
            return_full_text=False
        )
        return {"generated_text": answer, "contexts": [node.text for node in nodes_retrieved]}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during generation: {e}"
        )

@app.get("/")
def read_root():
    """
    Root endpoint to check API status.
    Returns:
        JSON response indicating API health.
    """
    return {"message": "RAG API is running successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
