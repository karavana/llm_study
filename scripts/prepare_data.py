import os
import uuid
import dill
from typing import List
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- CONFIGURATION ---
PDF_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dr_voss_diary.pdf')
MILVUS_DB_PATH = "milvus/milvus_data.db"
PARENT_DOCS_STORE_PATH = "milvus/parent_docs.pkl" # File to store parent chunks
COLLECTION_NAME = "veridia_retriever_chunks" 
EMBEDDING_MODEL_NAME = "snowflake/snowflake-arctic-embed-s"
EMBEDDING_DIM = 384  


def embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> List[List[float]]:
    """Generates embeddings for a list of texts."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()

def init_milvus_collection(client: MilvusClient, collection_name: str, dim: int):
    """Creates a new Milvus collection with a schema for the ParentDocumentRetriever."""
    if client.has_collection(collection_name):
        print(f"[!] Collection '{collection_name}' exists. Dropping...")
        client.drop_collection(collection_name)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    # This field will store the ID of the parent document
    schema.add_field("doc_id", DataType.VARCHAR, max_length=36) 
    schema.add_field("text", DataType.VARCHAR, max_length=2000)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

    client.create_collection(collection_name=collection_name, schema=schema, consistency_level="Strong")
    print(f"[✓] Collection '{collection_name}' created.")
    
def insert_child_docs(client: MilvusClient, collection_name: str, child_docs: List[Document]):
    """Embeds and inserts child documents with their parent's ID into Milvus."""
    # Extract text for embedding
    texts_to_embed = [doc.page_content for doc in child_docs]
    print(f"[*] Generating embeddings for {len(texts_to_embed)} child documents...")
    embeddings = embed_texts(texts_to_embed)
    
    # Prepare records with text, embedding, and parent doc_id from metadata
    records = [
        {
            "doc_id": doc.metadata['doc_id'],
            "text": doc.page_content,
            "embedding": vec
        } 
        for doc, vec in zip(child_docs, embeddings)
    ]
    
    client.insert(collection_name=collection_name, data=records)
    print(f"[✓] Inserted {len(records)} child documents into Milvus.")

def create_milvus_index(client: MilvusClient, collection_name: str):
    """Creates an index on the vector field."""
    print("[*] Creating index on the vector field...")
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="L2")
    client.create_index(collection_name=collection_name, index_params=index_params)
    print("[✓] Index created successfully.")


def main():
    """Main data preparation pipeline for the ParentDocumentRetriever."""
    # 1. Initialize Milvus Lite client
    client = MilvusClient(MILVUS_DB_PATH)
    init_milvus_collection(client, COLLECTION_NAME, EMBEDDING_DIM)

    # 2. Load the raw documents from PDF
    print(f"[*] Loading document from: {PDF_PATH}")
    loader = PyMuPDFLoader(PDF_PATH)
    raw_docs = loader.load()

    # 3. Define the parent and child splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    # 4. Generate and link parent/child documents
    print("[*] Generating parent and child chunks...")
    parent_docs = []
    child_docs = []
    
    # Split the raw documents into large "parent" chunks
    parent_chunks = parent_splitter.split_documents(raw_docs)

    for parent_chunk in parent_chunks:
        # Assign a unique ID to each parent chunk
        doc_id = str(uuid.uuid4())
        parent_chunk.metadata['doc_id'] = doc_id
        parent_docs.append(parent_chunk)
        
        # Split the parent chunk into smaller "child" chunks
        sub_docs = child_splitter.split_documents([parent_chunk])
        
        # Add the parent's ID to each child's metadata
        for sub_doc in sub_docs:
            sub_doc.metadata['doc_id'] = doc_id
        
        child_docs.extend(sub_docs)

    print(f"[✓] Created {len(parent_docs)} parent chunks and {len(child_docs)} child chunks.")

    # 5. Save the parent documents to disk for the API to use
    print(f"[*] Saving {len(parent_docs)} parent documents to '{PARENT_DOCS_STORE_PATH}'...")
    with open(PARENT_DOCS_STORE_PATH, "wb") as f:
        dill.dump(parent_docs, f)

    # 6. Insert the CHILD documents into Milvus
    insert_child_docs(client, COLLECTION_NAME, child_docs)
    
    # 7. Create the index for efficient searching
    create_milvus_index(client, COLLECTION_NAME)

    print("\n[✅] Parent-Child data preparation pipeline completed successfully!")

if __name__ == "__main__":
    main()