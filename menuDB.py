from llama_index.core import VectorStoreIndex
from qdrant_client import QdrantClient
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

class MenuDB:
    def __init__(self, qdrant_url):
        self._qdrant_url = qdrant_url
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)

    def create_index(self, collection_name, documents, service_context):
        try:
            vector_store = QdrantVectorStore(client=self._client, collection_name=collection_name)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)
            print("Knowledgebase created successfully!")
            return index
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
            return None