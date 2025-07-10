from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# === CONFIGURATION ===
MILVUS_COLLECTION = "veridia_chunks"
EMBEDDING_MODEL = "snowflake/snowflake-arctic-embed-s"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # küçük bir LLM önerisi
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5  # Milvus'tan kaç benzer chunk çekilecek

# === INIT FASTAPI ===
app = FastAPI(title="RAG System for Veridia")

# === MODELS ===
class QueryRequest(BaseModel):
    question: str

# === LOAD MODELS ON STARTUP ===
@app.on_event("startup")
def load_models():
    global embed_model, llm_tokenizer, llm_model, milvus_collection

    # Connect Milvus
    connections.connect("default", host="milvus", port="19530")
    milvus_collection = Collection(MILVUS_COLLECTION)
    milvus_collection.load()

    # Embedding model
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # LLM
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL).to(DEVICE)

    print("[✓] All models and DB connections are ready.")

# === UTILITIES ===
def embed_question(question: str) -> List[float]:
    return embed_model.encode([question], convert_to_numpy=True).tolist()[0]

def retrieve_context(query_embedding: List[float]) -> List[str]:
    search_params = {
        "data": [query_embedding],
        "anns_field": "embedding",
        "param": {"metric_type": "L2", "params": {"nprobe": 10}},
        "limit": TOP_K,
        "output_fields": ["text"],
    }
    results = milvus_collection.search(**search_params)
    return [hit.entity.get("text") for hit in results[0]]

def generate_answer(context_chunks: List[str], question: str) -> str:
    context_text = "\n".join(context_chunks)
    prompt = f"""You are an expert on the world of Veridia.

Context:
{context_text}

Question: {question}
Answer:"""

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = llm_model.generate(**inputs, max_new_tokens=300, do_sample=True)
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

# === ENDPOINT ===
@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        q_embed = embed_question(request.question)
        context_chunks = retrieve_context(q_embed)
        answer = generate_answer(context_chunks, request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
