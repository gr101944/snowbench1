def get_gpt_response(client, model_name):
    # Check and initialize 'messages' in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Trim the conversation history to manage token and message limits
    st.session_state.messages = trim_conversation_history(st.session_state.messages)

    # Getting the response from OpenAI
    openai_response = client.chat.completions.create(
        model=model_name,
        messages=st.session_state.messages
    )

    # Extracting the latest response message
    new_message = openai_response.choices[0].message

    # Update session state with the new response
    st.session_state.messages.append(new_message)

    return new_message