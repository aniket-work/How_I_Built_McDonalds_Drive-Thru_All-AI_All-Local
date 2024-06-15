import json
from llama_index.core import SimpleDirectoryReader
from menuDB import MenuDB
from reasoning import Reasoning
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()


class Assistant:
    """
    The Assistant class is responsible for managing the interaction with a menu database
    and reasoning engine, providing an interface for querying a language model.

    Attributes:
        _qdrant_url (str): URL for the Qdrant vector database.
        _menu_file (str): Path to the menu file.
        _collection_name (str): Name of the collection in the vector database.
        _prompt (str): Prompt for the reasoning engine.
        _menu_db (MenuDB): Instance of the MenuDB class for database operations.
        _reasoning (Reasoning): Instance of the Reasoning class for language model interaction.
        _index: Index created from the menu documents.
    """

    def __init__(self):
        """
        Initializes the Assistant class by loading configuration parameters from a JSON file,
        setting up the menu database, and creating an index for menu documents. Also, initializes
        the reasoning engine with the required service context.
        """
        # Load configuration parameters from a JSON file
        with open('config/config.json', 'r') as config_file:
            config = json.load(config_file)

        self._qdrant_url = config['qdrant_url']
        self._menu_file = config['menu_file']
        self._collection_name = config['collection_name']
        self._prompt = config['prompt']

        # Initialize the menu database
        self._menu_db = MenuDB(self._qdrant_url)
        self._reasoning = Reasoning(os.getenv("GROQ_API_KEY"), self._prompt)

        # Read and load menu documents
        reader = SimpleDirectoryReader(input_files=[self._menu_file])
        documents = reader.load_data()

        # Create an index for the menu documents
        self._index = self._menu_db.create_index(self._collection_name, documents, self._reasoning._service_context)

        # Initialize the reasoning engine's chat functionality
        self._reasoning.create_chat_engine(self._index)

    def interact_with_llm(self, customer_query):
        """
        Interacts with the language model using the provided customer query and returns the response.

        Args:
            customer_query (str): The query from the customer to interact with the language model.

        Returns:
            str: The response from the language model.
        """
        answer = self._reasoning.interact_with_llm(customer_query)
        return answer
