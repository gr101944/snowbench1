from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
import streamlit as st
def process_huggingface(question, llm):
    print ('In process_hugging_face')
    import json


    from langchain.prompts.prompt import PromptTemplate

    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {chat_history}
    Human: {input}
    AI Assistant:"""

    PROMPT = PromptTemplate(input_variables=["chat_history", "input"], template=template)
    
    repo_id = "tiiuae/falcon-7b-instruct" 
    

    conversationHF = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=False,
        memory=st.session_state['memory_hf'] 
    )
    response = conversationHF.predict(input=question)


    if response is None: 

        response= ("Sorry, no response found")

    jsondata = {
        "source": "Hugging Face",
        "response": response
    }

    json_data = json.dumps(jsondata)  
    return json_data