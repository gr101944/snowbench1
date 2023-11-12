import os

import boto3
import json
import dotenv
import openai
import PyPDF2
import io


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.vectorstores import Pinecone
import pinecone

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
    
    
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
bing_search_url = os.getenv('BING_SEARCH_URL')
bing_subscription_key = os.getenv('BING_SUBSCRIPTION_KEY')
bing_api = os.getenv('BARD_API')

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    print("OpenAI API key not found. Please make sure you have set the OPENAI_API_KEY environment variable.")

if SERPAPI_API_KEY:
    serpapi_api_key = SERPAPI_API_KEY
else:
    print("SERPAPI_API_KEY key not found. Please make sure you have set the SERPAPI_API_KEY environment variable.")

pinecone.init (
    api_key = os.environ['PINECONE_API_KEY'],
    environment = os.environ['PINECONE_ENVIRONMENT']    
)


memory = ConversationBufferMemory(return_messages=True)

def create_sidebar (st):
    # Need to push to env

    source_data_list = [ 'KR1']
    source_data_list_default = [ 'KR1']
    task_list = [ 'Data Load', 'Query']
    source_category = ['KR1']
    other_sources = ['Open AI', 'YouTube', 'Google',  'Bing', 'KR']
    other_sources_default = ['KR']
    embedding_options = ['OpenAI Ada', 'Other1']
    persistence_options = [ 'Vector DB', 'On Disk']
    # 
    url = "https://www.persistent.com/"
   
    st.sidebar.write(f"🔗 [**:green[Peristent]** ]({url})")    
   
    with st.sidebar.expander(" 🛠️ Configurations ", expanded=False):
    # Option to preview memory store
        model_name  = st.selectbox(label='LLM', options=['gpt-3.5-turbo','text-davinci-003'], help='GPT-4 in waiting list 🤨')
        embedding_model_name  = st.radio('Embedding', options=embedding_options, help='Option to change embedding model, keep in mind to match with the LLM 🤨')
        persistence_choice = st.radio('Persistence', persistence_options, help = "FAISS for on disk and Pinecone for Vector")
        
        k_similarity = st.text_input ("K value",type="default",value="5")
        max_output_tokens = st.text_input ("Max Output Tokens",type="default",value="512")
        print("k_similarity ", k_similarity )
        print("output_tokens ", max_output_tokens )
        print ('persistence_choice ', persistence_choice)    
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
    print ("***********task***************")
    print (task)
    if (task == 'Data Load'):
            print ('Data Load triggered, assiging ingestion source')
            ingest_source_chosen = st.radio('Choose :', source_data_list, help = "For loading documents into each KR specific FAISS index")
            uploaded_files = st.file_uploader('Upload Files Here', accept_multiple_files=True)
            upload_kr_docs_button = st.button("Upload", key="upload_kr_docs")
    elif (task == 'Query'):
            print ('In query')
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
                print ('case 1')
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
    print (ingest_source_chosen)
    return model_name, persistence_choice, selected_sources, uploaded_files , summarize, website_url, youtube_url,k_similarity, sources_chosen, task, upload_kr_docs_button, ingest_source_chosen, source_data_list, source_category

st.set_page_config(
    page_title="NexgenBot", 
    page_icon=":arrow_forward:"
    )


model_name, persistence_choice, selected_sources, uploaded_files, summarize, website_url, youtube_url,  k_similarity, sources_chosen, task, upload_kr_docs_button, ingest_source_chosen, source_data_list, source_category = create_sidebar (st)
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
if 'generated_hf' not in st.session_state:
    st.session_state['generated_hf'] = []
if 'entity_memory' not in st.session_state:
    st.session_state['entity_memory'] = ConversationEntityMemory (llm=llm, k=10)
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
    

def search_vector_store3 (persistence_choice, VectorStore, user_input, model, source, k_similarity):
    print ('In search_vector_store3', model)

    if 'chat_history_upload' not in st.session_state:
        st.session_state['chat_history_upload'] = []
        
    st.session_state['chat_history_upload'].append (user_input)
    template = """You are a chatbot having a conversation with a human.

        Given the following extracted parts of a long document and a question, explain the answer in a paragraph

        {context}

        {chat_history_upload}
        Human: {user_input}
        Chatbot:"""

    
    n = 4  # Replace with the desired value of n
    chat_history_upload = st.session_state['chat_history_upload']
    
    total_elements = len(chat_history_upload)
 
    result = ' '.join(chat_history_upload[:n]) if n <= total_elements else ' '.join(chat_history_upload)
    print ('Before similarity search, maintaining history for similarity search')
    print (result)
      
    docs = VectorStore.similarity_search(query=result, k=int (k_similarity))
    prompt = PromptTemplate(
            input_variables=["chat_history_upload", "user_input","context"], template=template
    )
    
    llm = OpenAI(model_name=model, temperature=0.7, max_tokens=2048)
   
    memory = ConversationBufferMemory(memory_key="chat_history_upload", input_key="user_input")

    chain = load_qa_chain(
            llm=llm, chain_type="stuff", memory=memory , prompt=prompt
        )
 
    response  = chain({"input_documents": docs, "user_input": user_input}, return_only_outputs=True)

    if "I don't know" in response or response == None:
        print ('In I dont know')
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









text_splitter = CharacterTextSplitter(        
    
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
text_splitter2 = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap  = 10,
    length_function = len,
    add_start_index = True,
)
text_splitter3 = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
def split_text_chunks(input_text, x):
    return [input_text[i:i+x] for i in range(0, len(input_text), x)]

def split_text_chunks2(input_text, x, percentage_overlap):
    overlap = int(x * (percentage_overlap / 100))
    chunks = []

    start_index = 0
    while start_index < len(input_text):
        end_index = start_index + x
        if end_index > len(input_text):
            end_index = len(input_text)

        chunk = input_text[start_index:end_index]
        chunks.append(chunk)

        start_index += x - overlap

    return chunks

def process_pdf_file(file_path):
    print ("in process_pdf_file")
    print ("file path: ", file_path)
    chunk_size = 400
    overlap = 20
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    aws_bucket = os.getenv('S3_BUCKET_NAME')
    print('Processing PDF, extracting chunks ... ', file_path)
    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)


    # Download PDF file from S3
    response = s3.get_object(Bucket=aws_bucket, Key=file_path)

    pdf_content = response["Body"].read()
    pdf_stream = io.BytesIO(pdf_content)

    # Create a PDF reader
    pdf_reader = PyPDF2.PdfReader(pdf_stream)

    # Extract text from each page
    text_content = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text_content.append(page.extract_text())
    print ('got text content:')
    print (text_content)
       # text = updf.readPDF_text_pdfReader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
        chunk_size = chunk_size ,
        chunk_overlap  = overlap,
        length_function = len,
    )
    chunks = text_splitter.create_documents(text_content)
    print ("Number of Chunks")
    print (len (chunks))
    # print (chunks)

    #chunks = split_text_chunks2 (text, chunk_size, percentage_overlap )

    if len(chunks) == 0:
        print ('No chunks extracted')
        return 0
    else:
        print (f'Number of chunks: {len(chunks)}')
    return chunks

def process_text_file(file_path):
    print('Processing text')
    chunk_size = 400
    overlap = 20
    print('Processing PDF, extracting chunks ... ', file_path)
    with open(file_path) as f:
        text_file= f.read()
   
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
        chunk_size = chunk_size ,
        chunk_overlap  = overlap,
        length_function = len,
    )
    total_chunks = text_splitter.create_documents([text_file])


    
    if len (total_chunks) > 2000:
        print ('Too many chunks' ,len(total_chunks))
        return
    if len(total_chunks) == 0:
        print ('No chunks extracted')
        return 0
    else:
        print (f'Number of chucks: {len(total_chunks)}')
        print (total_chunks[0])
    return total_chunks

def process_csv_file(file_path):
    print('Processing CSV')
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    for doc in docs:
        page_content = doc.page_content
        metadata = doc.metadata
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
    chunks = text_splitter.split_text(text=page_content)
    if len(chunks) == 0:
        print ('No chunks extracted')
        return 0
    else:
        print (f'Number of chuncks: {len(chunks)}')
    return chunks

def process_excel_file(file_path):
    print('Processing Excel')
    loader = UnstructuredExcelLoader(file_path, mode="elements")
    docs = loader.load()
    for doc in docs:
        page_content = doc.page_content
        metadata = doc.metadata
    chunks = text_splitter.split_text(text=page_content)
    if len(chunks) == 0:
        print ('No chunks extracted')
        return 0
    else:
        print (f'Number of chuncks: {len(chunks)}')
    return chunks

def process_pptx_file(file_path):
    print('Processing PPTX')
    loader = UnstructuredPowerPointLoader(file_path)
    docs = loader.load()
    for doc in docs:
        page_content = doc.page_content
        metadata = doc.metadata
    chunks = text_splitter.split_text(text=page_content)
    if len(chunks) == 0:
        print ('No chunks extracted')
        return 0
    else:
        print (f'Number of chuncks: {len(chunks)}')
    return chunks


def process_bing_search(prompt,llm):
    print ('In process Bing*************************', prompt)
   
    
    import json
    bing_search = BingSearchAPIWrapper(k=2)
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
    print ('In process Google*************************', prompt)
 
    
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

    print (data)

    json_data = json.dumps(data)  
    return json_data

def extract_youtube_id(youtube_url):
    return youtube_url.split('v=')[-1]

def process_YTLinks(youtube_video_url, user_input):
    print ('In process_YTLinks New', user_input)      

    youtube_video_id = extract_youtube_id (youtube_video_url)
    print('YouTube Id extraction')
    print (youtube_video_id)

    loader = YoutubeLoader.from_youtube_url(youtube_video_url, add_video_info=False)
    
    youtube_transcript_list = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
    chunks = text_splitter.split_documents(youtube_transcript_list)
     
    dotenv.load_dotenv(".env")
    env_vars = dotenv.dotenv_values()
    for key in env_vars:
        os.environ[key] = env_vars[key]
    pinecone.init (
        api_key = os.environ['PINECONE_API_KEY'],
        environment = os.environ['PINECONE_ENVIRONMENT']    
    )
    pinecone_index_name = os.environ['PINECONE_INDEX_NAME']
    print ('pinecone index name ', pinecone_index_name)
    
 
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
                                #print(docs[0].page_content[:200])
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

    print ('In process wiki new*************************')
    
   
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
    print (wiki_research)
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
    print ("result " , result)
  
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
    print ("Question... " + question)
    
    import json
    if 'chat_history_bard' not in st.session_state:
            st.session_state['chat_history_bard'] = []
    st.session_state['chat_history_bard'].append (question)
    result = ' '.join(st.session_state['chat_history_bard'])
    print ("result " , result)
    response = Bard().get_answer(result)
    print (response)

    # Extract the English response
    english_response = extract_english_response(response)
    print(english_response)


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

    # llama_repo = "meta-llama/Llama-2-70b-hf"
    # falcon_repo = "tiiuae/falcon-7b-instruct"
    # new_model_repo = "tiiuae/falcon-40b"

    PROMPT = PromptTemplate(input_variables=["chat_history", "input"], template=template)
    print ("PROMPT ", PROMPT)
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
    # print ("formatted_prompt ", formatted_prompt)

    # data = conversation.predict (input = formatted_prompt)
    # print (data)
    # print ("Type is ", type(data))
    # ai_assistance_response = data[0]['response']
    # print ( "ai_assistance_response ", ai_assistance_response)
    # assistant_instruction = ai_assistance_response.split('AI Assistant:', 1)[-1].strip()
    # print ( "assistant_instruction ", assistant_instruction)
    

    if response is None: 

        response= ("Sorry, no response found")

    jsondata = {
        "source": "Hugging Face",
        "response": response
    }

    json_data = json.dumps(jsondata)  
    return json_data



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
    print (s3_target_path)

    # Upload the file to S3
    s3.put_object(Body=bytes_data, Bucket=aws_bucket, Key=s3_target_path)
    print ("File uploaded in S3")

    # _ is used to ignore the base name of the file
    _, file_extension = os.path.splitext(uploaded_file.name)
        

    if file_extension.lower() == '.pdf': 
        chunks = process_pdf_file(s3_target_path)
    elif file_extension.lower() == '.txt':
        chunks = process_text_file(uploaded_file.name)
    elif file_extension.lower() == '.csv':
        chunks = process_csv_file(uploaded_file.name)
    elif file_extension.lower() == '.docx':
        chunks = process_docx_file(uploaded_file.name)
    elif file_extension.lower() == '.xlsx' or file_extension.lower() == '.xls':
        chunks = process_excel_file(uploaded_file.name)
    elif file_extension.lower() == '.pptx':
        chunks = process_pptx_file(uploaded_file.name)
    else:
        raise ValueError(f'Unsupported Filetype: {file_extension}')

    return chunks
def process_chatGPT2(prompt, model, Conversation):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    response = Conversation.run(input=prompt)
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
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
    print ('In process_knowledge_base ')
    from langchain.embeddings.openai import OpenAIEmbeddings

    model_name = 'text-embedding-ada-002'
    dotenv.load_dotenv(".env")
    env_vars = dotenv.dotenv_values()
    for key in env_vars:
        os.environ[key] = env_vars[key]
    pinecone.init (
        api_key = os.environ['PINECONE_API_KEY'],
        environment = os.environ['PINECONE_ENVIRONMENT']    
    )
    pinecone_index_name = os.environ['PINECONE_INDEX_NAME']

    
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
    print (type (uploaded_files))
    notFound_response = "The application was unable to extract content from the uploaded file."
    if len(uploaded_files) > 0:
        print(f'Number of files uploaded: {len(uploaded_files)}')
        docs_chunks = []  # Initialize docs_chunks list
        vector_stores = []

  

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
                api_key = os.environ['PINECONE_API_KEY'],
                environment = os.environ['PINECONE_ENVIRONMENT']    
            )
            pinecone_index_name = os.environ['PINECONE_INDEX_NAME']
            try:
                index_name = pinecone_index_name  # Specify the index name as a string
                indexPinecone = pinecone.Index(index_name)
                print('Deleting all vectors in Pinecone')
                indexPinecone.delete(delete_all=True)
                print('Deleted all vectors in Pinecone')

            except pinecone.exceptions.PineconeException as e:
                print(f"An error occurred: {str(e)}")
            print('***********docs_chunks**********')
            print (docs_chunks)

        
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
        'Open AI': process_chatGPT2,
        'Google':process_google_search,
        'Bing':process_bing_search,
        'Hugging Face':process_hugging_face2,
        'Website': process_bing_search,
        'YouTube': process_YTLinks,

        
    }

    responses = []
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
                print ('str_response : ', str_response)
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
                #with st.spinner("Crawling..."): 
                #    primary_urls = [website_url]
                #    depth = 2
                #    crawl_site(primary_urls, depth)
                str_response = selected_elements_functions[element]("site:"+ website_url + " " + prompt, llm)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            elif (element == 'YouTube'):
                print ('Processing YouTube')
                #with st.spinner("Crawling..."): 
                #    primary_urls = [website_url]
                #    depth = 2
                #    crawl_site(primary_urls, depth)
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
            print ('accumulated_json')
            print (accumulated_json)
      

            accumulated_json_str = json.dumps(accumulated_json)


    return accumulated_json_str


def get_response(user_input, source_data_list):
    import json
    print ('In get_response...')
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    if  user_input and len(selected_sources) > 0 and goButton:

        with st.spinner("Searching requested sources..."):        
            str_resp = selected_data_sources(selected_sources, user_input, uploaded_files, model, llm, Conversation, website_url, sources_chosen, source_data_list)               
            data = json.loads(str_resp)['all_responses']
            print ('data')
            print (data)

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
                    print ('in if')
                    response_dict[f"{source.lower().replace(' ', '_')}_response"] = response['response']
                    print ('reached 1')
           
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
                print (youtube_response)
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
                print (kr_response) 
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

            
            if model_name == "GPT-3.5":
                cost = total_tokens * 0.002 / 1000
            else:
                cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

          

            if st.session_state['all_response_dict']:
                with response_container:
                    #for i in range(len(st.session_state['all_response_dict'])):
                    #    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                    #    message(st.session_state["all_response_dict"][i], key=str(i))
                    download_str = []

                    latest_index = len(st.session_state['generated']) - 1

                    st.info(st.session_state["past"][latest_index], icon="🎸")
                    st.success(st.session_state["generated"][latest_index], icon="✅")
                    download_str.append(st.session_state["past"][latest_index])
                    download_str.append(st.session_state["generated"][latest_index])
                    if likeButton:
                        st.balloons()
                    if dislikeButton:
                        st.snow()
                    st.toast ("Developed by :orange[Rajesh Ghosh] 🎸")

                                    
                    if summarize:
                        summary_dict = []
                        st.subheader('Summary from all sources')
                        generated_string = str(st.session_state['generated'][-1])

                        print ('Generated String before summary')
                        print (generated_string)
                        summary = process_chatGPT2("Please generate a short summary of the following text in professional words: " + generated_string, model, Conversation)
            

                        summary_json = json.loads(summary)  # Parse the summary string as JSON
                        response = summary_json['response']  # Extract the 'response' field from the summary JSON

                        summary_dict.append(response)
                        
                        message(summary_dict)


response_container = st.container()
# container for text box
container = st.container()

with container:
    if (task =='Data Load'):
        if upload_kr_docs_button:
            uploaded_kr = process_uploaded_file( uploaded_files,  persistence_choice, ingest_source_chosen)
   
            st.write ("Done!")
            
            
    if (task =='Query'):
        selected_sources_text = ", ".join(selected_sources)
        ask_text = "**Selected Sources:** " + "**:green[" + selected_sources_text + "]**" 
    
    #st.markdown("This text is :red[colored red], and this is **:blue[colored]** and bold.")
        user_input = st.text_input(ask_text, key='input', value = '', placeholder='Ask something...', label_visibility="visible") 
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        goButton = col1.button("Go")
        likeButton = col7.button(":+1:")
        dislikeButton = col8.button(":-1:")
    
        if likeButton:
            st.balloons()
        if dislikeButton:
            st.snow()
        
        get_response (user_input, source_data_list)

    # Allow to download as wellst.session_state["past"]
        download_str = []
        # Display the conversation history using an expander, and allow the user to download it
        with st.expander("Download Conversation", expanded=False):
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                st.info(st.session_state["past"][i],icon="🎸")
                st.success(st.session_state["generated"][i], icon="✅")
                download_str.append(st.session_state["past"][i])
                download_str.append(st.session_state["generated"][i])
            
            # Can throw error - requires fix
            download_str = '\n'.join(download_str)
            if download_str:
                st.download_button('Download',download_str)
    
    




