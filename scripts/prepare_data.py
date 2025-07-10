import os
import fitz  # PyMuPDF
from typing import List
from sentence_transformers import SentenceTransformer
from pymilvus import connections, MilvusClient, utility, Collection, CollectionSchema, FieldSchema, DataType

PDF_PATH = "data/dr_voss_diary.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "veridia_chunks"
EMBEDDING_DIM = 384 #snowflake-arctic-embed-s model produces embeddings of dimension 384


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    with fitz.open(pdf_path) as doc:
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




def init_milvus_schema(collection_name: str, dim: int) -> MilvusClient:
    # Start Milvus Lite
    client = MilvusClient("milvus_data.db")

    if client.has_collection(collection_name):
        print(f"[!] Collection '{collection_name}' exists. Dropping...")
        client.drop_collection(collection_name)

    # Create schema
    schema = MilvusClient.create_schema()
    schema.add_field("chunk_id", DataType.INT64, is_primary=True, auto_id=True, description="Chunk ID")
    schema.add_field("text", DataType.VARCHAR, max_length=2000, enable_analyzer=True, enable_match=True, description="Chunk text")
    schema.add_field("text_dense_vector", DataType.FLOAT_VECTOR, dim=dim, description="Embedding vector")
    
    # Optional BM25 function (bonus)
    # from pymilvus import Function, FunctionType
    # bm25 = Function(
    #     name="text_bm25",
    #     input_field_names=["text"],
    #     output_field_names=["text_sparse_vector"],
    #     function_type=FunctionType.BM25,
    # )
    # schema.add_field("text_sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    # schema.add_function(bm25)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        consistency_level="Strong",
        enable_dynamic_field=False,
        overwrite=True
    )

    return client


def insert_chunks(client: MilvusClient, collection_name: str, chunks: List[str], embeddings: List[List[float]]):
    records = [{"text": t, "text_dense_vector": vec} for t, vec in zip(chunks, embeddings)]
    client.insert(collection_name=collection_name, data=records)
    print(f"[✓] Inserted {len(records)} chunks.")




def main():
    print("[*] Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("[*] Chunking text...")
    chunks = chunk_text(text)

    print(f"[*] Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    print("[*] Initializing Milvus and inserting embeddings...")
    client = init_milvus_schema(COLLECTION_NAME, EMBEDDING_DIM)
    insert_chunks(client, COLLECTION_NAME, chunks, embeddings)

    print("[✅] Data preparation pipeline completed.")


if __name__ == "__main__":
    main()
