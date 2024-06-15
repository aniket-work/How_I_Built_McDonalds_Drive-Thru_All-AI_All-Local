from llama_index.core import VectorStoreIndex
from qdrant_client import QdrantClient
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

class MenuDB:
    """
    A class to manage the creation of a vector store index for a menu database using Qdrant.

    Attributes:
        _qdrant_url (str): The URL for the Qdrant client.
        _client (QdrantClient): The Qdrant client instance used to interact with the Qdrant vector database.
    """

    def __init__(self, qdrant_url):
        """
        Initializes the MenuDB class with the specified Qdrant URL.

        Args:
            qdrant_url (str): The URL for the Qdrant client.
        """
        self._qdrant_url = qdrant_url
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)

    def create_index(self, collection_name, documents, service_context):
        """
        Creates a vector store index from the provided documents using the Qdrant vector database.

        Args:
            collection_name (str): The name of the collection in the Qdrant vector database.
            documents (list): A list of documents to be indexed.
            service_context: The service context required by the VectorStoreIndex.

        Returns:
            VectorStoreIndex: The created vector store index if successful.
            None: If there was an error during the index creation process.
        """
        try:
            # Initialize the Qdrant vector store
            vector_store = QdrantVectorStore(client=self._client, collection_name=collection_name)
            # Create a storage context with the vector store
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Create a vector store index from the documents
            index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)
            print("Knowledgebase created successfully!")
            return index
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
            return None
