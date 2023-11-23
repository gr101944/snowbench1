import os
import boto3
import json
import dotenv
import openai
import PyPDF2
import io
import uuid

import tiktoken
from streamlit_option_menu import option_menu


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.vectorstores import Pinecone
import pinecone


chunk_size = 400
chunk_overlap = 20


from langchain.chains import LLMChain 
from langchain.agents import load_tools

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor


from bardapi import Bard

import streamlit as st
from streamlit_chat import message

from langchain.document_loaders import UnstructuredFileLoader

from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter 

from langchain.document_loaders import YoutubeLoader
from langchain.utilities import BingSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain import HuggingFaceHub
from langchain.utilities import SerpAPIWrapper

dotenv.load_dotenv(".env")
env_vars = dotenv.dotenv_values()
for key in env_vars:
    os.environ[key] = env_vars[key]

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY=os.getenv('SERPAPI_API_KEY')

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION')

STATIC_ASSEST_BUCKET_URL = os.getenv("STATIC_ASSEST_BUCKET_URL")
STATIC_ASSEST_BUCKET_FOLDER = os.getenv("STATIC_ASSEST_BUCKET_FOLDER")
LOGO_NAME = os.getenv("LOGO_NAME")
    
    
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
BING_SEARCH_URL = os.getenv('BING_SEARCH_URL')
BING_SUBSCRIPTION_KEY = os.getenv('BING_SUBSCRIPTION_KEY')
PROMPT_INSERT_LAMBDA = os.getenv('PROMPT_INSERT_LAMBDA')
PROMPT_UPDATE_LAMBDA = os.getenv('PROMPT_UPDATE_LAMBDA')
PROMPT_QUERY_LAMBDA = os.getenv('PROMPT_QUERY_LAMBDA')
lambda_client = boto3.client('lambda', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)


if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    print("OpenAI API key not found. Please make sure you have set the OPENAI_API_KEY environment variable.")

if SERPAPI_API_KEY:
    serpapi_api_key = SERPAPI_API_KEY
else:
    print("SERPAPI_API_KEY key not found. Please make sure you have set the SERPAPI_API_KEY environment variable.")

pinecone.init (
    api_key = os.getenv('PINECONE_API_KEY'),
    environment = os.getenv('PINECONE_ENVIRONMENT')    
)


memory = ConversationBufferMemory(return_messages=True)
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def create_sidebar (st):
    # Need to push to env

    st.sidebar.markdown(
        """
        <style>
            img {
                margin-top: -90px;  /* Adjust this value as needed */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    image_path = os.path.join(STATIC_ASSEST_BUCKET_URL, STATIC_ASSEST_BUCKET_FOLDER, LOGO_NAME)
    with st.sidebar:
        st.image(image_path, width=55)
    

    source_data_list = [ 'KR1']
    source_data_list_default = [ 'KR1']
    task_list = [ 'Data Load', 'Query']
    source_category = ['KR1']
    other_sources = ['Open AI', 'YouTube', 'Google', 'KR', 'text2Image']
    text2Image_source = ['text2Image']
    other_sources_default = ['KR']
    embedding_options = ['text-embedding-ada-002']
    persistence_options = [ 'Vector DB']
    # 
    url = "https://www.persistent.com/"
    
   
    with st.sidebar.expander(" 🛠️ Configurations ", expanded=False):
    # Option to preview memory store
        model_name  = st.selectbox(label='LLM', options=['gpt-3.5-turbo','text-davinci-003'], help='GPT-4 in waiting list 🤨')
        embedding_model_name  = st.radio('Embedding', options=embedding_options, help='Option to change embedding model, keep in mind to match with the LLM 🤨')
        persistence_choice = st.radio('Persistence', persistence_options, help = "Using Pinecone...")
        
        k_similarity_text = st.text_input ("K value",type="default",value="5")
        k_similarity = int(k_similarity_text)
        max_output_tokens = st.text_input ("Max Output Tokens",type="default",value="512")
        print("k_similarity ", k_similarity )
        print("max_output_tokens ", max_output_tokens )
        print ('persistence_choice ', persistence_choice)    
        print ('embedding_model_name ', embedding_model_name)   
    task= None
    task = st.sidebar.radio('Choose task:', task_list, help = "Program can both Load Data and perform query", index=0)
    selected_sources = None
    summarize = None
    website_url=  None
    youtube_url = None
    uploaded_files = None
    upload_kr_docs_button = None
    ingest_source_chosen = None
    sources_chosen = None
    selected_sources_image = None
    print ("***********task***************")
    print (task)
    if (task == 'Data Load'):
            print ('Data Load triggered, assiging ingestion source')
            ingest_source_chosen = st.radio('Choose :', source_data_list, help = "For loading documents into each KR specific FAISS index")
            uploaded_files = st.file_uploader('Upload Files Here', accept_multiple_files=True)
            upload_kr_docs_button = st.button("Upload", key="upload_kr_docs")
    elif (task == 'Query'):
            print ('In Query')
            selected_sources_image = st.sidebar.multiselect(
                'Image Generation:',
                text2Image_source
               
              )
            if 'text2Image' in selected_sources_image:
                selected_sources = []
            else:    
            # sources_chosen = st.sidebar.multiselect( 'KR:',source_data_list, source_data_list_default )
                selected_sources = st.sidebar.multiselect(
                    'Sources:',
                    other_sources,
                    other_sources_default
                )
                if 'KR' in selected_sources:
                    # Show the 'sources_chosen' multiselect
                    sources_chosen = st.sidebar.multiselect(
                        'KR:',
                        source_data_list,
                        source_data_list_default
                    )
                else:
                    # 'KR' is not selected, so don't show the 'sources_chosen' multiselect
                    sources_chosen = None
            

                
            
                if len(selected_sources) > 1:
                    summarize = st.sidebar.checkbox("Summarize", value=False, key=None, help='If checked, summarizes content from all sources along with individual responses', on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
                else:
                
                    summarize = None
            
                if 'Website' in selected_sources:
                    print ('website selected in sidebar')
                    website_url = st.sidebar.text_input("Website Url:", key='url', value = '', placeholder='Type URL...', label_visibility="visible") 
                else:
                    website_url = None

                if 'YouTube' in selected_sources:
                    print ('YouTube selected in sidebar')
                    youtube_url = st.sidebar.text_input("YouTube Url:", key='url', value = '', placeholder='Paste YouTube URL...', label_visibility="visible") 
                else:
                    youtube_url = None
    
    return (
        model_name,
        persistence_choice,
        selected_sources,
        uploaded_files,
        summarize,
        website_url,
        youtube_url,
        k_similarity,
        sources_chosen,
        task,
        upload_kr_docs_button,
        ingest_source_chosen,
        source_data_list,
        source_category,
        embedding_model_name,
        selected_sources_image
    )




(
    model_name,
    persistence_choice,
    selected_sources,
    uploaded_files,
    summarize,
    website_url,
    youtube_url,
    k_similarity,
    sources_chosen,
    task,
    upload_kr_docs_button,
    ingest_source_chosen,
    source_data_list,
    source_category,
    embedding_model_name,
    selected_sources_image
) = create_sidebar(st)

print ("ingest_source_chosen ", ingest_source_chosen)
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-3.5-turbo"
llm = OpenAI(model_name=model, temperature=0.7, max_tokens=512)
# Initialise session state variables
if 'generated_wiki' not in st.session_state:
    st.session_state['generated_wiki'] = []
    

if 'all_response_dict' not in st.session_state:
    st.session_state['all_response_dict'] = []   
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = "Rajesh5"  
    

if 'generated_hf' not in st.session_state:
    st.session_state['generated_hf'] = []
if 'entity_memory' not in st.session_state:
    st.session_state['entity_memory'] = ConversationEntityMemory (llm=llm, k=k_similarity)
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
if 'generated_bing' not in st.session_state:
    st.session_state['generated_bing'] = []    
if 'generated_website' not in st.session_state:
    st.session_state['generated_website'] = []
if 'generated_uploads' not in st.session_state:
    st.session_state['generated_uploads'] = []
if 'sel_source' not in st.session_state:
    st.session_state['sel_source'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state["past"] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = []


counter_placeholder = st.sidebar.empty()
clear_button = None
#counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
if (task == 'Query'):
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

Conversation = ConversationChain(llm=llm, prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, memory=st.session_state['entity_memory'] )

table_name = os.environ['PROCESS_TABLE']

def process_text2image (prompt):
    import requests
    from datetime import datetime
    text2image_model_name = "dall-e-3"
   
    # user_prompt = "Generate a high-resolution, realistic image of a Mahindra Thar vehicle in a full and front-facing view. The scene should be captured as if photographed with a Canon high-quality camera. The backdrop should showcase majestic mountains, providing a picturesque setting. Pay attention to details such as lighting, reflections, and shadows to ensure a lifelike representation of the vehicle in this scenic environment."
    image_sizes = ["1024x1024", "1024x1792", "1792x1024"]
    # st.write ("Generating image...")
    response = openai.Image.create(
        prompt=prompt,
        model=text2image_model_name,
        n=1,
        size=image_sizes[0],
        quality="standard", 
    )

    if 'data' in response and response['data']:
        item = response['data'][0]  # Assuming you want to generate only the first image
        image_url = item['url']
        print (image_url)
        file_name = "image" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".png"
        print (image_url)
            # Create an S3 client
        aws_region = os.getenv('AWS_DEFAULT_REGION')
        aws_bucket = os.getenv('S3_PUBLIC_ACCESS')
        aws_bucket_input_path = os.getenv('S3_PUBLIC_ACCESS_PATH')
        s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
        image_data = requests.get(image_url).content
        s3.put_object(Body=image_data, Bucket=aws_bucket, Key=aws_bucket_input_path + "/" + file_name)

        st.image(image_url, caption="Generated Image", use_column_width=True)
    else:
        st.write("No data available to generate an image.")
    
def process_Wikipedia2(prompt, llm):
    print("Processing Wikipedia2...", prompt)
    wiki_not_found = "No good Wikipedia Search Result was found in wikipedia"  
    wikipedia = WikipediaAPIWrapper()
    tools = load_tools (["wikipedia"], llm=llm)
    agent = initialize_agent (tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    wiki_research = agent.run(prompt) 

    if (wiki_research == wiki_not_found):
        return_code = 400        
    else:
        return_code = 200
    
        # Create a dictionary to hold the data
    data = {
            "source": "Wikipedia",
            "response": wiki_research,
            "return_code": return_code
    }
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)

    return json_data


def clear_session():
    st.session_state['generated'] = []
    st.session_state['chat_history_upload'] = [] 
    
    st.session_state['memory_hf'] = []
    st.session_state['generated_KR'] = []
    
    st.session_state['generated_youtube'] = []
    st.session_state['all_response_dict'] = []
    st.session_state['chat_history_bard'] = []    
    st.session_state['chat_history_wiki'] = []    
    st.session_state['sel_source'] = []
    st.session_state['generated_bing'] = []    
    st.session_state['memory'] = []
    st.session_state['entity_memory'] = []
    st.session_state['generated_wiki'] = []
    st.session_state['generated_openai'] = []
    st.session_state['generated_google'] = []
    st.session_state['generated_bard'] = []
    st.session_state['generated_website'] = []
    st.session_state['generated_uploads'] = []
    st.session_state['generated_hf'] = []
    st.session_state["past"] = []
    st.session_state["memory_wiki"] = []

    
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['generated_KR'] = []
    st.session_state['memory_hf'] = []    
    st.session_state['generated_youtube'] = []
    st.session_state['chat_history_upload'] = []   
    st.session_state["memory_wiki"] = []
    st.session_state['chat_history_bard'] = [] 
    st.session_state['chat_history_wiki'] = []  
    st.session_state['generated_wiki'] = []
    st.session_state['generated_bing'] = []
    st.session_state['all_response_dict'] = []
    st.session_state['generated_bard'] = []
    st.session_state['memory'] = []
    st.session_state['generated_openai'] = []
    st.session_state['generated_google'] = []
    st.session_state['generated_uploads'] = []
    st.session_state['generated_hf'] = []
    st.session_state['generated_website'] = []
    st.session_state["past"] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

def process_CloudAssets(prompt, llm):
    print("Processing Cloud Assets...")

def process_GoogleDocs(prompt, llm):
    print("Processing Google Docs...")
    
def create_text_splitter(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)
    

def search_vector_store3 (persistence_choice, VectorStore, user_input, model, source, k_similarity):
    print ('In search_vector_store3', model)
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks import get_openai_callback

    if 'chat_history_upload' not in st.session_state:
        st.session_state['chat_history_upload'] = []
        
    st.session_state['chat_history_upload'].append (user_input)
    
    chat_history_upload = st.session_state['chat_history_upload']
   
    template = """You are a chatbot having a conversation with a human.

        Given the following extracted parts of a long document and a question, explain the answer in a paragraph

        {context}

        {chat_history_upload}
        Human: {user_input}
        Chatbot:"""
    
    n = 4  # Replace with the desired value of n
    
    
    total_elements = len(chat_history_upload)
 
    result = ' '.join(chat_history_upload[:n]) if n <= total_elements else ' '.join(chat_history_upload)

      
    docs = VectorStore.similarity_search(query=result, k=int (k_similarity))
    prompt = PromptTemplate(
            input_variables=["chat_history_upload", "user_input", "context"], template=template
    )
   
    memory = ConversationBufferMemory(memory_key="chat_history_upload", input_key="user_input")

    chain = load_qa_chain(
            llm=llm, chain_type="stuff", memory=memory , prompt=prompt
        )
    with get_openai_callback() as cb:
        response  = chain({"input_documents": docs, "user_input": user_input}, return_only_outputs=True)
        input_tokens = cb.prompt_tokens
        output_tokens = cb.completion_tokens
        cost = round(cb.total_cost, 6)
        random_string = str(uuid.uuid4())
        
        user_name_logged = "Rajesh5"
        data = {    
            "userName": user_name_logged,
            "promptName": "prompt-" + random_string,
            "prompt": user_input,
            "completion": response['output_text'],
            "summary": "summary-from-code",
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cost": cost,
            "like": ""
        }
        if 'curent_user' not in st.session_state:
            st.session_state['current_user'] = []
        if 'curent_promptName' not in st.session_state:
                st.session_state['curent_promptName'] = []
        st.session_state['current_user'] = user_name_logged
        st.session_state['curent_promptName'] = "prompt-" + random_string

        # Invoke the Lambda function
        lambda_function_name = PROMPT_INSERT_LAMBDA
        lambda_response = lambda_client.invoke(
            FunctionName=lambda_function_name,
            InvocationType='RequestResponse',  # Use 'Event' for asynchronous invocation
            Payload=json.dumps(data)
        )
        if lambda_response['StatusCode'] != 200:
            raise Exception(f"AWS Lambda invocation failed with status code: {lambda_response['StatusCode']}")
        else:
            print ("Success!")
        st.sidebar.write(f"**Usage Info:** ")
        st.sidebar.write(f"**Input Tokens:** {input_tokens}")
        st.sidebar.write(f"**Output Tokens:** {output_tokens}")
        st.sidebar.write(f"**Cost($):** {cost}")
        # st.write(f"**Input Tokens:** {input_tokens} | **Output Tokens:** {output_tokens} | **Cost:** {cost}")


    if "I don't know" in response or response == None:
       
        st.write('Info not in artefact...') 
    else:
        print ('returning response from upload...')

    data = {
            "source": source,
            "response": response
    }
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)

    return json_data

def process_csv_file(file_path):
    print('Processing CSV')
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    for doc in docs:
        page_content = doc.page_content
        metadata = doc.metadata
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_text(text=page_content)
    if len(chunks) == 0:
        print ('No chunks extracted')
        return 0
    else:
        print (f'Number of chuncks: {len(chunks)}')
    return chunks

def process_docx_file(file_path):
    print('Processing DOCX')
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    for doc in docs:
        page_content = doc.page_content
        metadata = doc.metadata
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_text(text=page_content)
    if len(chunks) == 0:
        print ('No chunks extracted')
        return 0
    else:
        print (f'Number of chuncks: {len(chunks)}')
    return chunks

def process_pptx_file(file_path):
    print('Processing PPTX')
    chunks = "dumy chunks"
    loader = UnstructuredPowerPointLoader(file_path)
    docs = loader.load()
    for doc in docs:
        page_content = doc.page_content
        metadata = doc.metadata
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_text(text=page_content)
    if len(chunks) == 0:
        print ('No chunks extracted')
        return 0
    else:
        print (f'Number of chuncks: {len(chunks)}')
    return chunks

def process_bing_search(prompt,llm):
    print ('In process Bing', prompt)
    
    import json
    bing_search = BingSearchAPIWrapper(k=k_similarity)
    tools = [
        Tool(
            name = "bing_search",
            func=bing_search.run,
            description="useful for when you need to answer questions about current events"
        )
    ]
    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    promptAgent = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    if 'memory_bing' not in st.session_state:
        st.session_state['memory_bing'] = ConversationBufferMemory(memory_key="chat_history")    
   
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=promptAgent)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=st.session_state['memory_bing'] 
    )
    #agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    response = agent_chain.run(input=prompt)
    if response is None:
        response = "Sorry, no data found!"        

    if 'site:' in prompt:
        source_mod = "Website"
    else:
        source_mod = "Bing"     
   
    data = {
        "source": source_mod,
        "response": response
    }

    json_data = json.dumps(data)  
    return json_data

def process_google_search(prompt,llm):
    print ('In process Google', prompt)
 
    
    import json
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        )
    ]
    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    promptAgent = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    if 'memory_google' not in st.session_state:
        st.session_state['memory_google'] = ConversationBufferMemory(memory_key="chat_history")    
   
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=promptAgent)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=st.session_state['memory_google'] 
    )
    #agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    response = agent_chain.run(input=prompt)
    if response is None:
        response = "Sorry, no data found!"        
    #agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    #response = agent.run(prompt)
    if 'site:' in prompt:
        source_mod = "Website"
    else:
        source_mod = "Google"     
   
    data = {
        "source": source_mod,
        "response": response
    }

  

    json_data = json.dumps(data)  
    return json_data

def extract_youtube_id(youtube_url):
    return youtube_url.split('v=')[-1]

def process_YTLinks(youtube_video_url, user_input):
    print ('In process_YTLinks New', user_input)      

    youtube_video_id = extract_youtube_id (youtube_video_url)
    loader = YoutubeLoader.from_youtube_url(youtube_video_url, add_video_info=False)
    
    youtube_transcript_list = loader.load()
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_documents(youtube_transcript_list)
     
    dotenv.load_dotenv(".env")
    env_vars = dotenv.dotenv_values()
    for key in env_vars:
        os.environ[key] = env_vars[key]
    pinecone.init (
        api_key = os.getenv('PINECONE_API_KEY'),
        environment = os.getenv('PINECONE_ENVIRONMENT')   
    )
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')

    
 
    if len (chunks) < 100:

        if len(chunks) == 0:
            print("No chunks extracted from: ", youtube_video_url)                    
        else:
        
            try:
                index_name = pinecone_index_name  # Specify the index name as a string
                indexPinecone = pinecone.Index(index_name)
                print('Deleting all vectors in Pinecone')
                indexPinecone.delete(delete_all=True)
                print('Deleted all vectors in Pinecone')

            except pinecone.exceptions.PineconeException as e:
                print(f"An error occurred: {str(e)}")

            embeddings = OpenAIEmbeddings()        
            docsearch = Pinecone.from_texts([t.page_content for t in chunks], embeddings, index_name=index_name)
            persistence_choice = "Vector DB"
            resp = search_vector_store3 (persistence_choice, docsearch, user_input, model, "YouTube", k_similarity)
       
            data_dict = json.loads(resp)
            # Extracting the 'output_text' from the dictionary
            output_text = data_dict['response']['output_text']

            data = {
                "source": "YouTube",
                "response": output_text
            }
            json_data = json.dumps(data)

            return json_data
    else:
        print ("too many chunks")

def process_Wikipedia2(prompt, llm):
    print("Processing Wikipedia2...", prompt)
    wiki_not_found = "No good Wikipedia Search Result was found in wikipedia"  
    wikipedia = WikipediaAPIWrapper()
    tools = load_tools (["wikipedia"], llm=llm)
    agent = initialize_agent (tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    wiki_research = agent.run(prompt) 

    if (wiki_research == wiki_not_found):
        return_code = 400        
    else:
        return_code = 200
    
        # Create a dictionary to hold the data
    data = {
            "source": "Wikipedia",
            "response": wiki_research,
            "return_code": return_code
    }
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)

    return json_data 

def process_wiki_search_new(prompt,llm):

    print ('In process wiki new')
    
   
    import json
    wiki_not_found = "No good Wikipedia Search Result was found in wikipedia"  
       
    search = WikipediaAPIWrapper()

    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        )
    ]

    prefix = """Have a conversation with a human, answering the following questions in a short paragraph. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    """

    promptAgent = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history"],
    )
    if 'memory_wiki' not in st.session_state:
        st.session_state['memory_wiki'] = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=llm, prompt=promptAgent)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=st.session_state['memory_wiki'] 
    )
    #agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    wiki_research = agent_chain.run(input=prompt)

    if (wiki_research == wiki_not_found):
        return_code = 400        
    else:
        return_code = 200
    
        # Create a dictionary to hold the data
    data = {
            "source": "Wikipedia",
            "response": wiki_research,
            "return_code": return_code
    }
    json_data = json.dumps(data)  
    return json_data

def trim_user_word(string):
    string = string.rstrip()  # Remove trailing whitespace, including newline
    if string.endswith('User'):
        return string[:-len('User')].rstrip()
    else:
        return string
    
def remove_repetitions(response):
    import re
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)  # Split into sentences
    unique_sentences = list(set(sentences))  # Remove duplicates
    unique_response = ' '.join(unique_sentences)
    return unique_response
 
def process_hugging_face(question):
    import json
    if 'chat_history_huggingface' not in st.session_state:
            st.session_state['chat_history_huggingface'] = []
    st.session_state['chat_history_huggingface'].append (question)
    result = ' '.join(st.session_state['chat_history_huggingface'])
    
    repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    falcon_llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 300}
    )
    # --------------------------------------------------------------
    # Create a PromptTemplate and LLMChain
    # --------------------------------------------------------------
    template = """
    You are an artificial intelligence assistant. The assistant gives helpful, summarised, and polite answers to the user's questions.
    {question}
 
    """
    
    prompt = PromptTemplate(template=template, input_variables=["question"] + st.session_state['chat_history_huggingface'])
  
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

# --------------------------------------------------------------
# Run the LLMChain
# --------------------------------------------------------------
    
    response = trim_user_word (llm_chain.run(question))

    response = remove_repetitions(response)

    data = {
        "source": "Hugging Face",
        "response": response
    }

    json_data = json.dumps(data)  
    return json_data

def extract_english_response(response_dict):
    import re
    from langdetect import detect
    # Extract the content from the dictionary
    content = response_dict.get('content', '')

    # Use regular expressions to find sentences
    sentences = re.findall(r'[A-Za-z][^.!?]*[.!?]', content)

    # Filter only the English sentences
    english_sentences = [sentence for sentence in sentences if detect(sentence) == 'en']

    # Combine the English sentences into a single string
    english_response = ' '.join(english_sentences)

    return english_response.strip()

def process_bard(question):
    print ("Process Bard...")    
    
    import json
    if 'chat_history_bard' not in st.session_state:
            st.session_state['chat_history_bard'] = []
    st.session_state['chat_history_bard'].append (question)
    result = ' '.join(st.session_state['chat_history_bard'])

    response = Bard().get_answer(result)
 
    english_response = extract_english_response(response)
  
    data = {
        "source": "Bard",
        "response": english_response
    }

    json_data = json.dumps(data)  
    return json_data

def process_hugging_face2(question):
    print ('In process_hugging_face2')
    import json
    if 'memory_hf' not in st.session_state:
        st.session_state['memory_hf'] = ConversationBufferMemory(memory_key="chat_history")

    from langchain.prompts.prompt import PromptTemplate

    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {chat_history}
    Human: {input}
    AI Assistant:"""

    PROMPT = PromptTemplate(input_variables=["chat_history", "input"], template=template)
    
    repo_id = "tiiuae/falcon-7b-instruct" 
    
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
    )
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=False,
        memory=st.session_state['memory_hf'] 
    )
    response = conversation.predict(input=question)


    if response is None: 

        response= ("Sorry, no response found")

    jsondata = {
        "source": "Hugging Face",
        "response": response
    }

    json_data = json.dumps(jsondata)  
    return json_data


def process_pdf_file_new(file_content):
    pdf_stream = io.BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    text_content = [page.extract_text() for page in pdf_reader.pages]


    text_splitter = create_text_splitter(chunk_size, chunk_overlap)

    chunks = text_splitter.create_documents(text_content)

    return chunks

def process_text_file_new(file_content):
    print ("In process_file_new")
    text_content = file_content.decode('utf-8')
    print ("text_content")
    print ("-------------")
    print (text_content)

    text_splitter = create_text_splitter(chunk_size, chunk_overlap)

    chunks = text_splitter.create_documents([text_content])
    print (chunks)
    return chunks
def process_xlsx_file(file_content):
    print('Processing Excel')
    chunks = "chunks dummy"
    # loader = UnstructuredExcelLoader(file_path, mode="elements")
    # docs = loader.load()
    # for doc in docs:
    #     page_content = doc.page_content
    #     metadata = doc.metadata
    # chunks = text_splitter.split_text(text=page_content)
    # if len(chunks) == 0:
    #     print ('No chunks extracted')
    #     return 0
    # else:
    #     print (f'Number of chuncks: {len(chunks)}')
    return chunks

def process_file(file_path):
    print(f'Rajesh New function Processing file: {file_path}')

    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    aws_bucket = os.getenv('S3_BUCKET_NAME')

    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)

    response = s3.get_object(Bucket=aws_bucket, Key=file_path)
    file_content = b""

    for chunk in response["Body"].iter_chunks():
        file_content += chunk

    file_extension = os.path.splitext(file_path)[1][1:].lower()  # Get file extension without the dot

    if file_extension == 'pdf':
        chunks = process_pdf_file_new(file_content)

    elif file_extension == 'txt':
        chunks = process_text_file_new(file_content)
        
    elif file_extension == 'csv':
        chunks = process_csv_file(file_content)
        
    elif file_extension == 'xlsx':
        chunks = process_xlsx_file(file_content)

    else:
        # Handle other file formats if needed
        print(f'Unsupported file format: {file_extension}')
        return

    if len(chunks) > 2000:
        print(f'Too many chunks: {len(chunks)}')
        return

    if len(chunks) == 0:
        print('No chunks extracted')
        return 0

    print(f'Number of chunks: {len(chunks)}')
    print(chunks[0])
    return chunks




def extract_chunks_from_uploaded_file(uploaded_file):
    print('In extract_chunks_from_uploaded_file')
    bytes_data = uploaded_file.read()
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    aws_bucket = os.getenv('S3_BUCKET_NAME')
    aws_bucket_input_path = os.getenv('S3_BUCKET_INPUT_PATH')


    # Create an S3 client
    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)

    # Define the target path in S3
    s3_target_path = aws_bucket_input_path + uploaded_file.name
    print ("s3_target_path ", s3_target_path)
    
    
    # Upload the file to S3
    s3.put_object(Body=bytes_data, Bucket=aws_bucket, Key=s3_target_path)
    

    # _ is used to ignore the base name of the file
    _, file_extension = os.path.splitext(uploaded_file.name)
        

    if file_extension.lower() == '.pdf': 
        chunks = process_file(s3_target_path)
        print ("pdf_chunks:££££ ", len(chunks))
        # chunks = process_pdf_file(s3_target_path)
        # print ("    chunks:££££ ", len(chunks))
    elif file_extension.lower() == '.txt':
        print ("Processing .txt ***************************")
        
        chunks = process_file(s3_target_path)
        print ("txt_chunks:££££ ", len(chunks))
    elif file_extension.lower() == '.csv':
        chunks = process_csv_file(uploaded_file.name)
    elif file_extension.lower() == '.docx':
        chunks = process_docx_file(uploaded_file.name)
    elif file_extension.lower() == '.xlsx' or file_extension.lower() == '.xls':
        chunks = process_xlsx_file(uploaded_file.name)
    elif file_extension.lower() == '.pptx':
        chunks = process_pptx_file(uploaded_file.name)
    else:
        raise ValueError(f'Unsupported Filetype: {file_extension}')

    return chunks

def process_openai(prompt, model, Conversation):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    response = Conversation.run(input=prompt)
    st.session_state['messages'].append({"role": "assistant", "content": response})
 
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    
    response_openai = {
        "source": "Open AI",
        "response": response,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }
    # Convert the dictionary to a JSON string
    json_data = json.dumps(response_openai)
 
    return json_data

def process_knowledge_base(prompt, model, Conversation, sources_chosen, source_data_list):
    print ('In process_knowledge_base Rajesh to change to get embed model from config ')
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    model_name = embedding_model_name
    dotenv.load_dotenv(".env")
    env_vars = dotenv.dotenv_values()
    for key in env_vars:
        os.environ[key] = env_vars[key]
    pinecone.init (
        api_key = os.getenv('PINECONE_API_KEY'),
        environment = os.getenv('PINECONE_ENVIRONMENT')   
    )
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')

    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    text_field = "text"

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )
    index_name = pinecone_index_name
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )
    
    resp = search_vector_store3 (persistence_choice, vectorstore, prompt, model, "KR", 5)

    return resp
    
def process_uploaded_file(uploaded_files,  persistence_choice, ingest_source_chosen):
    import json
    print('in process_uploaded_file')
  
  
    if len(uploaded_files) > 0:
        print(f'Number of files uploaded: {len(uploaded_files)}')
        docs_chunks = []  # Initialize docs_chunks list
  
        for index, uploaded_file in enumerate(uploaded_files):
            if len(uploaded_file.name) >= 5 and uploaded_file.name[-5] == '.' and uploaded_file.name[-4:].isalpha():
                store_name = uploaded_file.name[:-5] 
            elif len(uploaded_file.name) >= 4 and uploaded_file.name[-4] == '.' and uploaded_file.name[-3:].isalpha():
                store_name = uploaded_file.name[:-4] 
            else:
                print ('Check input file extension')
                return
            print ('Creating chunks...')
            chunks = extract_chunks_from_uploaded_file(uploaded_file)

            if chunks == 0:
                print("No chunks extracted from: ", uploaded_file)                    
            else:    
                print(f'Processing file {index + 1}: {uploaded_file.name}')
                st.write(f'Processing file {index + 1}: {uploaded_file.name}')            
                docs_chunks.extend(chunks) 
                print (f'Total number of chunks : {len(docs_chunks)}')
                    
        embeddings = OpenAIEmbeddings()

        if ingest_source_chosen in source_data_list:
            print(f'processing {ingest_source_chosen} KR')
            # Dynamically get the file name based on the user's choice
            file_name = f"{ingest_source_chosen.lower()}_vector_file_name"
            print (f'Number of chunks in docs_chunks {len(docs_chunks)}')
            dotenv.load_dotenv(".env")
            env_vars = dotenv.dotenv_values()
            for key in env_vars:
                os.environ[key] = env_vars[key]
            pinecone.init (
                api_key = os.getenv('PINECONE_API_KEY'),
                environment = os.getenv('PINECONE_ENVIRONMENT')   
            )
            pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
            try:
                index_name = pinecone_index_name  # Specify the index name as a string
                indexPinecone = pinecone.Index(index_name)
                print('Deleting all vectors in Pinecone')
                indexPinecone.delete(delete_all=True)
                print('Deleted all vectors in Pinecone')

            except pinecone.exceptions.PineconeException as e:
                print(f"An error occurred: {str(e)}")
        
            docsearch = Pinecone.from_texts([t.page_content for t in docs_chunks], embeddings, index_name=index_name)
           
            return ingest_source_chosen

def selected_data_sources(selected_elements, prompt, uploaded_files, model, llm, Conversation, website_url, sources_chosen, source_data_list):
    print ("In selected_data_sources")
    import json

    all_responses = []
    selected_elements_functions = {
       
        'Wikipedia': process_Wikipedia2,
        'Cloud Assets': process_CloudAssets,
        'Google Docs': process_GoogleDocs,
        'Bard': process_bard,
        'Uploads': process_uploaded_file,
        'KR':process_knowledge_base,
        'Open AI': process_openai,
        'Google':process_google_search,
        'Bing':process_bing_search,
        'Hugging Face':process_hugging_face2,
        'Website': process_bing_search,
        'YouTube': process_YTLinks,

        
    }

    for element in selected_elements:
        if element in selected_elements_functions:
            if element == 'Uploads':
                str_response = selected_elements_functions[element](prompt, uploaded_files, persistence_choice)
           
                json_response = json.loads(str_response)
             
                all_responses.append(json_response)
            elif (element == 'Open AI'):
                str_response = selected_elements_functions[element](prompt, model, Conversation)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            elif (element == 'KR'):
                print ('Processing KR')
                str_response = selected_elements_functions[element](prompt, model, Conversation, sources_chosen, source_data_list)
            
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            elif (element == 'Google'):
                print ('Processing Google')
                str_response = selected_elements_functions[element](prompt, llm)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            elif (element == 'Bard'):
                print ('Processing Bard')
                str_response = selected_elements_functions[element](prompt)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            elif (element == 'Bing'):
                print ('Processing Bing')
                str_response = selected_elements_functions[element](prompt, llm)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            elif (element == 'Website'):
                print ('Processing Website ')
                str_response = selected_elements_functions[element]("site:"+ website_url + " " + prompt, llm)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            elif (element == 'YouTube'):
                print ('Processing YouTube')
                str_response = selected_elements_functions[element](youtube_url , prompt)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            elif (element == 'Hugging Face'):
                str_response = selected_elements_functions[element](prompt)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            elif (element == 'Wikipedia'):
                str_response = selected_elements_functions[element](prompt, llm)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            else:
                print ("check chosen sources")
            accumulated_json = {"all_responses": all_responses}

      

            accumulated_json_str = json.dumps(accumulated_json)


    return accumulated_json_str

def update_prompt(like_status):
    data = {
        "userName": st.session_state['current_user'],
        "promptName": st.session_state['curent_promptName'],
        "like": like_status
    }

    lambda_function_name = PROMPT_UPDATE_LAMBDA
    lambda_response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(data)
    )

    if lambda_response['StatusCode'] != 200:
        raise Exception(f"AWS Lambda invocation failed with status code: {lambda_response['StatusCode']}")

def get_response(user_input, source_data_list):
    import json
    print ('In get_response...')
    
    if 'text2Image' in selected_sources_image and user_input and goButton:
        with st.spinner("Generating image..."):
            with response_container:
                 process_text2image(user_input)
    else:  

        if  user_input and len(selected_sources) > 0 and goButton:
          print ("Go pressed")

          with st.spinner("Searching requested sources..."):        
            str_resp = selected_data_sources(selected_sources, user_input, uploaded_files, model, llm, Conversation, website_url, sources_chosen, source_data_list)               
            data = json.loads(str_resp)['all_responses']

            response_dict = {
                'wiki_response': None,
                'google_response': None,
                'bing_response': None,
                'openai_response': None,
                'hugging_face_response': None,
                'uploads_response': None,
                'bard_response':None,
                'youtube_response':None,
                'kr_response':None
            }
            for response in data:
                source = response['source']
                if source in selected_sources:
                  
                    response_dict[f"{source.lower().replace(' ', '_')}_response"] = response['response']
                 
           
            st.session_state["past"].append(user_input)
        
            st.session_state['sel_source'].append(selected_sources)


            # Assign responses to variables
            wiki_response = response_dict.get('wikipedia_response')
            bard_response = response_dict.get('bard_response')
            kr_response = response_dict.get('kr_response')
            google_response = response_dict.get('google_response')
            website_response = response_dict.get('website_response')
            openai_response = response_dict.get('open_ai_response')
            huggingface_response = response_dict.get('hugging_face_response')
            uploads_response = response_dict.get('uploads_response')
            bing_response = response_dict.get('bing_response')
            youtube_response = response_dict.get('youtube_response')
            all_response_str = ''

            if wiki_response:
                st.session_state['generated_wiki'].append(wiki_response)
                #st.session_state['generated'].append(wiki_response)
                all_response_str = all_response_str + "From Wikipedia  \n" + "-----------------------------" + "\n\n" + wiki_response + "\n\n"

            if bard_response:
                print ('in bard response')
                #st.session_state['generated'].append(google_response)
                st.session_state['generated_bard'].append(bard_response)
                all_response_str = all_response_str + "From Bard  \n" + "-----------------------------" + "\n\n" + bard_response + "\n\n"

            if youtube_response:
                print ('in YouTube response')
               
                #st.session_state['generated'].append(google_response)
                st.session_state['generated_youtube'].append(youtube_response)
                all_response_str = all_response_str + "From YouTube  \n" + "-----------------------------" + "\n\n" + youtube_response + "\n\n"
            
            if google_response:
                print ('in google response')
                #st.session_state['generated'].append(google_response)
                st.session_state['generated_google'].append(google_response)
                all_response_str = all_response_str + "From Google  \n" + "-----------------------------" + "\n\n" + google_response + "\n\n"

            if bing_response:
                print ('in bing response')
                #st.session_state['generated'].append(google_response)
                st.session_state['generated_bing'].append(bing_response)
                all_response_str = all_response_str + "From Bing  \n" + "-----------------------------" + "\n\n" + bing_response + "\n\n"

            if website_response:
                print ('in website response')
                #st.session_state['generated'].append(website_response)
                st.session_state['generated_website'].append(website_response)
                all_response_str = all_response_str + "From Website  \n" + "-----------------------------" + "\n\n" + website_response + "\n\n"
             
            if openai_response:
                #st.session_state['generated'].append(openai_response)
                st.session_state['generated_openai'].append(openai_response)
                all_response_str = all_response_str + "From OpenAI  \n" + "----------------------" + "\n\n" + openai_response + "\n\n"
            if kr_response:
                print ('kr_response *****************************************') 
              
                #st.session_state['generated'].append(openai_response)
                st.session_state['generated_KR'].append(kr_response)
                choice_str = ', '.join(sources_chosen) if sources_chosen else "None selected"
                all_response_str = all_response_str + "From KR: " + choice_str +  "\n" + "----------------------" + "\n\n" + kr_response['output_text'] + "\n\n"
            if huggingface_response:
                #st.session_state['generated'].append(huggingface_response)
                st.session_state['generated_hf'].append(huggingface_response)
                all_response_str = all_response_str + "From Hugging Face  \n" + "-----------------------------" + "\n\n" + huggingface_response + "\n\n"

            if uploads_response:
                st.session_state['generated_uploads'].append(uploads_response['output_text'])
                #st.session_state['generated'].append(uploads_response['output_text'])
                all_response_str = all_response_str + "From Uploaded Docs  \n" + "-----------------------------" + "\n\n" + uploads_response['output_text'] + "\n\n"
              
            else:
                print("Uploads data is not available.")

           
            st.session_state["all_response_dict"].append (all_response_str)
            st.session_state['generated'].append(all_response_str)

            if st.session_state['all_response_dict']:
                with response_container:

                    download_str = []

                    latest_index = len(st.session_state['generated']) - 1

                    st.info(st.session_state["past"][latest_index], icon="✅")
                    st.success(st.session_state["generated"][latest_index], icon="✅")
                    download_str.append(st.session_state["past"][latest_index])
                    download_str.append(st.session_state["generated"][latest_index])
                                   
                    if summarize:
                        summary_dict = []
                        st.subheader('Summary from all sources')
                        generated_string = str(st.session_state['generated'][-1])

                        summary = process_openai("Please generate a short summary of the following text in professional words: " + generated_string, model, Conversation)

                        summary_json = json.loads(summary)  # Parse the summary string as JSON
                  
                        response = summary_json['response']  # Extract the 'response' field from the summary JSON
                     
                        summary_dict.append(response)
                        st.success(summary_dict[0], icon="✅")
                        

response_container = st.container()
# container for text box
container = st.container()

with container:    
    if (task =='Data Load'):
        if upload_kr_docs_button:
            uploaded_kr = process_uploaded_file( uploaded_files,  persistence_choice, ingest_source_chosen)
            st.write ("Done!")
             
    if (task =='Query'):
        
        if 'text2Image' in selected_sources_image:
            selected_sources_text = 'text2Image'
            default_text_value = "Generate a high-resolution image with the exceptional sharpness, clarity, and color fidelity characteristic of a professional photograph taken with a Canon EOS-1D C DSLR Camera. Ensure the image exhibits impeccable details, vibrant colors, and minimal noise, emulating the superior image quality of this renowned camera. Capture the essence of professional photography, with a focus on realistic textures, crisp edges, and accurate representation of lighting. The final output should resemble a masterpiece produced by the Canon EOS-1D C"
            placeholder_default = None
            ask_text = "**Selected Sources:** " + "**:green[" + selected_sources_text + "]**" 
            user_input = st.text_area(ask_text, height=150, key='input', value = default_text_value, placeholder=placeholder_default, label_visibility="visible") 
        else:
            selected_sources_text = ", ".join(selected_sources)
            default_text_value = ''
            placeholder_default = "Ask something..."
            ask_text = "**Selected Sources:** " + "**:green[" + selected_sources_text + "]**" 
            user_input = st.text_input(ask_text,  key='input', value = default_text_value, placeholder=placeholder_default, label_visibility="visible") 
        
        print ("new code")
        
        # col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        # goButton = col1.button("Go")
        # likeButton = col7.button(":+1:")
        # dislikeButton = col8.button(":-1:")
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 1, 1, 1, 1, 1])

        with col1:
            goButton = st.button("Go")

        with col7:
            likeButton = st.button(":+1:")

        with col8:
            dislikeButton = st.button(":-1:")

        if likeButton:   
            update_prompt("Yes")

        if dislikeButton: 
            update_prompt("No")

       
        get_response (user_input, source_data_list)
        download_str = []
        # Display the conversation history using an expander, and allow the user to download it
        with st.expander("Download Conversation", expanded=False):
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                st.info(st.session_state["past"][i],icon="✅")
                st.success(st.session_state["generated"][i], icon="✅")
                download_str.append(st.session_state["past"][i])
                download_str.append(st.session_state["generated"][i])
 
            download_str = '\n'.join(download_str)
            if download_str:
                st.download_button('Download',download_str)
        if st.button("Past Queries"):
            st.write("Getting raw data from dynamoDB...")
            st.write("Fetching date for:", st.session_state['current_user'])
            data = {
                "userName": st.session_state['current_user']
            }

            lambda_function_name = PROMPT_QUERY_LAMBDA
            lambda_response = lambda_client.invoke(
                FunctionName=lambda_function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(data)
            )
            response_payload = json.loads(lambda_response['Payload'].read().decode('utf-8'))
            print (response_payload)
            st.write("Done. Printed in the log console...")
            
            parsed_data = (response_payload)
 

           
           