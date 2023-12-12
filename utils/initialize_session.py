import streamlit as st
def initialize_session():
    if 'memory_hf' not in st.session_state:
        st.session_state['memory_hf'] = []
    if 'generated_wiki' not in st.session_state:
        st.session_state['generated_wiki'] = [] 
    if 'all_response_dict' not in st.session_state:
        st.session_state['all_response_dict'] = [] 
    if 'generated_hf' not in st.session_state:
        st.session_state['generated_hf'] = []
    if 'feedback_given' not in st.session_state:
        st.session_state['feedback_given'] = [] 
    if 'generated_openai' not in st.session_state:
        st.session_state['generated_openai'] = []
    if 'generated_KR' not in st.session_state:
        st.session_state['generated_KR'] = []
    if 'generated_youtube' not in st.session_state:
        st.session_state['generated_youtube'] = []    
    if 'generated_google' not in st.session_state:
        st.session_state['generated_google'] = []
    if 'generated_bard' not in st.session_state:
        st.session_state['generated_bard'] = []
    if 'generated_uploads' not in st.session_state:
        st.session_state['generated_uploads'] = []
    if 'sel_source' not in st.session_state:
        st.session_state['sel_source'] = []
    if 'generated_response' not in st.session_state:
        st.session_state['generated_response'] = []
    if 'user_prompts' not in st.session_state:
        st.session_state["user_prompts"] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'button_pressed' not in st.session_state:
        st.session_state['button_pressed'] = []    
    if 'cost' not in st.session_state:
        st.session_state['cost'] = []
    if 'total_tokens' not in st.session_state:
        st.session_state['total_tokens'] = []
    if 'total_cost' not in st.session_state:
        st.session_state['total_cost'] = 0.0
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = []