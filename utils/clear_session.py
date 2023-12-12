import streamlit as st
def clear_session():
    print ("clear_session")
    st.session_state['generated_response'] = []
    st.session_state['generated_KR'] = []
    st.session_state['memory_hf'] = []    
    st.session_state['generated_youtube'] = []
    st.session_state['chat_history_upload'] = []   
    st.session_state["memory_wiki"] = []
    st.session_state['chat_history_bard'] = [] 
    st.session_state['chat_history_wiki'] = []  
    st.session_state['generated_wiki'] = []
    st.session_state['all_response_dict'] = []
    st.session_state['generated_bard'] = []
    st.session_state['memory'] = []
    st.session_state['generated_openai'] = []
    st.session_state['generated_google'] = []
    st.session_state['generated_uploads'] = []
    st.session_state['generated_hf'] = []
    st.session_state["user_prompts"] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
  