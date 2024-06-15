import json

with open('config/model_config.json', 'r') as config_file:
    model_config = json.load(config_file)

CHAT_MODE = model_config['chat_mode']
TOKEN_LIMIT = model_config['token_limit']