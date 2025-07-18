from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from contextlib import asynccontextmanager


# === CONFIGURATION ===
MILVUS_COLLECTION = "veridia_retriever_chunks"
EMBEDDING_MODEL = "snowflake/snowflake-arctic-embed-s"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"meta-llama/Llama-3.2-1B" # I was rejected when I asked to use this model; Your request to access this repo has been rejected by the repo's authors.
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
    models["milvus_client"] = MilvusClient("milvus/milvus_data.db")
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
    return models["embed_model"].encode([question], convert_to_numpy=True).tolist()[0]

def retrieve_context(query_embedding: List[float]) -> List[str]:
    results = models["milvus_client"].search(
        collection_name=MILVUS_COLLECTION, 
        data=[query_embedding],
        limit=TOP_K,
        output_fields=["text"],
    )
    return [hit['entity']['text'] for hit in results[0]]

def generate_answer(context_chunks: List[str], question: str) -> str:
    context_text = "\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY A DIRECT QUOTE from the provided context. 
Do NOT summarize or paraphrase.
Return EXACTLY ONE and ONLY ONE "COMPLETE" sentence that directly answers the question.

Context:
{context_text}

Question: {question}
Answer:"""

    try:
        print("[DEBUG] Generating answer with prompt:")
        print(prompt)
        inputs = models["llm_tokenizer"](prompt, return_tensors="pt").to(DEVICE)

        outputs = models["llm_model"].generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.1,
            eos_token_id=models["llm_tokenizer"].eos_token_id
        )

        full_output = models["llm_tokenizer"].decode(outputs[0], skip_special_tokens=True)
        generated_text = full_output[len(prompt):].strip()
        first_line = generated_text.split("\n")[0].strip()
        return first_line

    except Exception as e:
        print("[ERROR in generate_answer]", e)
        raise


# === ENDPOINT ===
@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        print(f"[REQUEST] Question: {request.question}")
        q_embed = embed_question(request.question)
        print("[DEBUG] Got embedding")
        context_chunks = retrieve_context(q_embed)
        print("[DEBUG] Retrieved context (len={}):".format(len(context_chunks)))
        for idx, c in enumerate(context_chunks):
            print(f"[DEBUG] Chunk {idx+1}:\n{c[:200]}...")

        answer = generate_answer(context_chunks, request.question)
        print("[RESPONSE] Answer:", answer)
        return {"answer": answer}

    except Exception as e:
        import traceback
        print("[ERROR] Exception occurred:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
