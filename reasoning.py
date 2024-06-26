from llama_index.core import ServiceContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import json
from constants import CHAT_MODE, TOKEN_LIMIT

class Reasoning:
    """
    A class to handle reasoning and interaction with a language model using the Groq model and HuggingFace embeddings.

    Attributes:
        _llm (Groq): The language model instance.
        _service_context (ServiceContext): The service context for the language model.
        _prompt (str): The system prompt for the chat engine.
        _chat_engine: The chat engine initialized with the vector index and memory buffer.
    """

    def __init__(self, api_key, prompt):
        """
        Initializes the Reasoning class with the specified API key and prompt.

        Args:
            api_key (str): The API key for authenticating with the Groq language model.
            prompt (str): The system prompt for the chat engine.
        """
        with open('config/model_config.json', 'r') as config_file:
            model_config = json.load(config_file)

        self._llm = Groq(model=model_config['llm_model'], api_key=api_key)
        embed_model = HuggingFaceEmbedding(model_name=model_config['embed_model'])
        self._service_context = ServiceContext.from_defaults(llm=self._llm, embed_model=embed_model)
        self._prompt = prompt

    def create_chat_engine(self, index):
        """
        Creates a chat engine using the provided vector index.

        Args:
            index: The vector index to be used for the chat engine.

        Process:
            - Initializes a chat memory buffer with a token limit.
            - Creates a chat engine from the index with the specified chat mode, memory, and system prompt.
        """
        memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)
        self._chat_engine = index.as_chat_engine(
            chat_mode=CHAT_MODE,
            memory=memory,
            system_prompt=self._prompt,
        )

    def interact_with_llm(self, customer_query):
        """
        Interacts with the language model using the chat engine to generate a response to the customer query.

        Args:
            customer_query (str): The customer's query to be processed by the language model.

        Returns:
            str: The response generated by the language model.
        """
        AgentChatResponse = self._chat_engine.chat(customer_query)
        answer = AgentChatResponse.response
        return answer
