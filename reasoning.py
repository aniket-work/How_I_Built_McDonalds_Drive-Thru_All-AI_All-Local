from llama_index.core import ServiceContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import json
from constants import CHAT_MODE, TOKEN_LIMIT

class Reasoning:
    def __init__(self, api_key, prompt):
        with open('config/model_config.json', 'r') as config_file:
            model_config = json.load(config_file)

        self._llm = Groq(model=model_config['llm_model'], api_key=api_key)
        embed_model = HuggingFaceEmbedding(model_name=model_config['embed_model'])
        self._service_context = ServiceContext.from_defaults(llm=self._llm, embed_model=embed_model)
        self._prompt = prompt

    def create_chat_engine(self, index):
        memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)
        self._chat_engine = index.as_chat_engine(
            chat_mode=CHAT_MODE,
            memory=memory,
            system_prompt=self._prompt,
        )

    def interact_with_llm(self, customer_query):
        AgentChatResponse = self._chat_engine.chat(customer_query)
        answer = AgentChatResponse.response
        return answer