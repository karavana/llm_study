import os
import pymupdf
from typing import List
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType

# --- CONFIGURATION ---
PDF_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dr_voss_diary.pdf')
MILVUS_DB_PATH = "milvus/milvus_data.db" 
COLLECTION_NAME = "veridia_chunks"
EMBEDDING_DIM = 384  
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    with pymupdf.open(pdf_path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    return text

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits the text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def embed_chunks(chunks: List[str], model_name: str = "snowflake/snowflake-arctic-embed-s") -> List[List[float]]:
    """Generates embeddings for each chunk."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()

def init_milvus_collection(client: MilvusClient, collection_name: str, dim: int):
    """Creates a new Milvus collection with the specified schema."""
    if client.has_collection(collection_name):
        print(f"[!] Collection '{collection_name}' exists. Dropping...")
        client.drop_collection(collection_name)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True, description="Chunk ID")
    schema.add_field("text", DataType.VARCHAR, max_length=2000, description="Chunk text")
    schema.add_field("text_dense_vector", DataType.FLOAT_VECTOR, dim=dim, description="Embedding vector")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        consistency_level="Strong"
    )
    print(f"[✓] Collection '{collection_name}' created.")

def insert_chunks(client: MilvusClient, collection_name: str, chunks: List[str], embeddings: List[List[float]]):
    """Inserts chunks and their embeddings into the collection."""
    records = [{"text": t, "text_dense_vector": vec} for t, vec in zip(chunks, embeddings)]
    client.insert(collection_name=collection_name, data=records)
    print(f"[✓] Inserted {len(records)} chunks.")


def create_milvus_index(client: MilvusClient, collection_name: str):
    """Creates an index on the vector field for efficient searching."""
    print("[*] Creating index on the vector field...")
    
    # Define the index parameters
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="text_dense_vector",
        index_type="AUTOINDEX",  # Let Milvus choose the best index type
        metric_type="L2"      # L2 is standard for sentence embeddings
    )
    
    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    
    print("[✓] Index created successfully.")


def main():
    """Main data preparation pipeline."""
    # Start Milvus Lite client
    client = MilvusClient(MILVUS_DB_PATH)

    print("[*] Initializing Milvus collection...")
    init_milvus_collection(client, COLLECTION_NAME, EMBEDDING_DIM)

    print("[*] Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("[*] Chunking text...")
    chunks = chunk_text(text)

    print(f"[*] Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    print("[*] Inserting data into Milvus...")
    insert_chunks(client, COLLECTION_NAME, chunks, embeddings)

    # This is the final and necessary step for searching
    create_milvus_index(client, COLLECTION_NAME)

    print("\n[✅] Data preparation pipeline completed.")


if __name__ == "__main__":
    main()