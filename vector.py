from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os
import json

# ===========================================
# Load your merged transaction dataset
# ===========================================
df = pd.read_csv("marged_dataset.csv")   # <-- change filename if needed

# ===========================================
# Embeddings model
# ===========================================
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ===========================================
# Vector DB settings
# ===========================================
db_location = "./fraud_chroma_db"
add_documents = not os.path.exists(db_location)

# ===========================================
# Convert each transaction row -> Document
# ===========================================
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():

        # Convert entire row to JSON string
        row_json = row.to_dict()
        row_text = json.dumps(row_json)

        document = Document(
            page_content=row_text,  # entire transaction as text
            metadata={
                "transaction_id": row_json.get("TRANSACTION_ID", "data not available"),
                "amount": row_json.get("AMOUNT_AUTHORIZED", "data not available"),
                "date": row_json.get("ISO_TRX_LOCAL_DATE_AND_TIME", "data not available")
            }
        )

        documents.append(document)
        ids.append(str(i))

# ===========================================
# Create / load Chroma vector store
# ===========================================
vector_store = Chroma(
    collection_name="transaction_records",
    persist_directory=db_location,
    embedding_function=embeddings
)

# ===========================================
# Populate database if first time
# ===========================================
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# ===========================================
# Create retriever
# ===========================================
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}    # retrieve top 5 similar transactions
)

print("RAG retriever ready.")
