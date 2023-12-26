from utils.get_num_tokens_from_string import get_num_tokens_from_string
def trim_conversation_history(messages, max_messages=10, max_tokens=1000):
    """
    Trims the conversation history to stay within the message and token limits.

    Parameters:
    - messages (list): The conversation history.
    - max_messages (int): The maximum number of messages to retain.
    - max_tokens (int): The maximum number of tokens to retain.

    Returns:
    - list: The trimmed conversation history.
    """
    trimmed_messages = messages[-max_messages:]  # Keep only the last 'max_messages' messages

    total_tokens = 0
    for message in reversed(trimmed_messages):
        message_tokens = get_num_tokens_from_string(message['content'], 'gpt-3.5-turbo')
        if total_tokens + message_tokens > max_tokens:
            break
        total_tokens += message_tokens

    return trimmed_messages[-(len(trimmed_messages) - trimmed_messages.index(message)):]