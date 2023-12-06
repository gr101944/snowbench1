def get_max_input_tokens(model_name):
    model_tokens_mapping = {
        'gpt-3.5-turbo': 4096,
        'gpt-4': 8192,
        'gpt-4-1106-preview': 128000,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-16k': 16385
    }
    return model_tokens_mapping.get(model_name, None)