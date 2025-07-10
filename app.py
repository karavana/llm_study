from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from contextlib import asynccontextmanager

# === CONFIGURATION ===
MILVUS_COLLECTION = "veridia_chunks"
EMBEDDING_MODEL = "snowflake/snowflake-arctic-embed-s"
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"  # accessible & high-quality
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5  


# === MODELS ===
models = {}
class QueryRequest(BaseModel):
    question: str
    
# === LOAD MODELS ON STARTUP ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("[*] Loading models and starting Milvus Lite...")
    models["milvus_client"] = MilvusClient("milvus_data.db")
    models["embed_model"] = SentenceTransformer(EMBEDDING_MODEL)
    models["llm_tokenizer"] = AutoTokenizer.from_pretrained(LLM_MODEL)
    models["llm_model"] = AutoModelForCausalLM.from_pretrained(LLM_MODEL).to(DEVICE)
    print("[✓] All models loaded and Milvus Lite is running.")
    
    yield
    
    # Clean up models on shutdown
    print("[*] Clearing models...")
    models.clear()

# === INIT FASTAPI ===
app = FastAPI(title="RAG System for Veridia", lifespan=lifespan)


# === UTILITIES ===
def embed_question(question: str) -> List[float]:
    return embed_model.encode([question], convert_to_numpy=True).tolist()[0]

def retrieve_context(query_embedding: List[float]) -> List[str]:
    results = milvus_client.search(
        collection_name=MILVUS_COLLECTION, 
        data=[query_embedding],
        limit=TOP_K,
        output_fields=["text"],
    )
    return [hit['entity']['text'] for hit in results[0]]

def generate_answer(context_chunks: List[str], question: str) -> str:
    context_text = "\n".join(context_chunks)

    prompt = f"""You are an expert on the world of Veridia. Use the provided context to answer the question accurately and concisely.

Context:
{context_text}

Question: {question}
Answer:"""

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        eos_token_id=llm_tokenizer.eos_token_id
    )

    full_output = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to extract only the part after the original prompt
    generated_text = full_output[len(prompt):].strip()

    # Optional: return just the first line (short-form answer)
    first_line = generated_text.split("\n")[0].strip()

    return first_line


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
