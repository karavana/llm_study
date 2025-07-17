import os
import pymupdf # Still needed by the loader
from typing import List
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType

# --- LANGCHAIN INTEGRATION ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION (Adjusted for smarter chunking) ---
PDF_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dr_voss_diary.pdf')
MILVUS_DB_PATH = "milvus/milvus_data.db" 
COLLECTION_NAME = "veridia_chunks"
EMBEDDING_MODEL_NAME = "snowflake/snowflake-arctic-embed-s"
EMBEDDING_DIM = 384  
CHUNK_SIZE = 1200 
CHUNK_OVERLAP = 180 


def embed_chunks(chunks: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> List[List[float]]:
    """Generates embeddings for each chunk. (This function remains the same)."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()

def init_milvus_collection(client: MilvusClient, collection_name: str, dim: int):
    """Creates a new Milvus collection. (This function remains the same)."""
    if client.has_collection(collection_name):
        print(f"[!] Collection '{collection_name}' exists. Dropping...")
        client.drop_collection(collection_name)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    # Corrected schema field names for clarity and consistency
    schema.add_field("id", DataType.INT64, is_primary=True, description="Primary key")
    schema.add_field("text", DataType.VARCHAR, max_length=2000, description="Chunk text")
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim, description="Embedding vector")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        consistency_level="Strong"
    )
    print(f"[✓] Collection '{collection_name}' created.")

def insert_chunks(client: MilvusClient, collection_name: str, chunks: List[str], embeddings: List[List[float]]):
    """Inserts chunks and their embeddings into the collection. (This function remains the same)."""
    records = [{"text": t, "embedding": vec} for t, vec in zip(chunks, embeddings)]
    client.insert(collection_name=collection_name, data=records)
    print(f"[✓] Inserted {len(records)} chunks.")


def create_milvus_index(client: MilvusClient, collection_name: str):
    """Creates an index on the vector field. (This function remains the same)."""
    print("[*] Creating index on the vector field...")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="L2"
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    print("[✓] Index created successfully.")


def main():
    """Main data preparation pipeline using the Hybrid Approach."""
    # 1. Start Milvus Lite client (Your working manual code)
    client = MilvusClient(MILVUS_DB_PATH)

    # 2. Initialize Milvus collection (Your working manual code)
    print("[*] Initializing Milvus collection...")
    init_milvus_collection(client, COLLECTION_NAME, EMBEDDING_DIM)

    # 3. Load Document with LangChain (Replaces extract_text_from_pdf)
    print(f"[*] Loading document from: {PDF_PATH}")
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()

    # 4. Split Text with LangChain (Replaces your old chunk_text function)
    print("[*] Splitting text with LangChain's RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents)
    # We need a simple list of strings for our manual embedding function
    chunks = [doc.page_content for doc in split_docs]
    print(f"[✓] Created {len(chunks)} semantically-split chunks.")

    # 5. Generate Embeddings (Your working manual code)
    print(f"[*] Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    # 6. Insert Data into Milvus (Your working manual code)
    print("[*] Inserting data into Milvus...")
    insert_chunks(client, COLLECTION_NAME, chunks, embeddings)

    # 7. Create Index (Your working manual code)
    create_milvus_index(client, COLLECTION_NAME)

    print("\n[✅] Hybrid data preparation pipeline completed successfully!")


if __name__ == "__main__":
    main()