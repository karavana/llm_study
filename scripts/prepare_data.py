import os
import fitz  # PyMuPDF
from typing import List
from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

PDF_PATH = "data/dr_voss_diary.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "veridia_chunks"
EMBEDDING_DIM = 768  # depends on the model


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


def init_milvus(collection_name: str, dim: int):
    """Initializes connection and collection schema for Milvus."""
    connections.connect(alias="default", uri="sqlite://:@:")

    if utility.has_collection(collection_name):
        print(f"[!] Collection '{collection_name}' already exists. Dropping and recreating...")
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Chunks of Veridia diary")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    })
    collection.load()
    return collection


def insert_to_milvus(collection: Collection, chunks: List[str], embeddings: List[List[float]]):
    """Inserts chunk texts and their embeddings into Milvus."""
    entities = [chunks, embeddings]
    insert_result = collection.insert(entities)
    print(f"[✓] Inserted {len(insert_result.primary_keys)} records into Milvus.")


def main():
    print("[*] Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("[*] Chunking text...")
    chunks = chunk_text(text)

    print(f"[*] Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    print("[*] Initializing Milvus and inserting embeddings...")
    collection = init_milvus(COLLECTION_NAME, EMBEDDING_DIM)
    insert_to_milvus(collection, chunks, embeddings)

    print("[✅] Data preparation pipeline completed.")


if __name__ == "__main__":
    main()
