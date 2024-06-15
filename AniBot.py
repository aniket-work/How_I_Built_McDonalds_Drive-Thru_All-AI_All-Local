import json
from llama_index.core import SimpleDirectoryReader
from menuDB import MenuDB
from reasoning import Reasoning
from dotenv import load_dotenv
import os
load_dotenv()

class Assistant:
    def __init__(self):
        with open('config/config.json', 'r') as config_file:
            config = json.load(config_file)

        self._qdrant_url = config['qdrant_url']
        self._menu_file = config['menu_file']
        self._collection_name = config['collection_name']
        self._prompt = config['prompt']

        self._menu_db = MenuDB(self._qdrant_url)
        self._reasoning = Reasoning(os.getenv("GROQ_API_KEY"), self._prompt)

        reader = SimpleDirectoryReader(input_files=[self._menu_file])
        documents = reader.load_data()
        self._index = self._menu_db.create_index(self._collection_name, documents, self._reasoning._service_context)
        self._reasoning.create_chat_engine(self._index)

    def interact_with_llm(self, customer_query):
        answer = self._reasoning.interact_with_llm(customer_query)
        return answer