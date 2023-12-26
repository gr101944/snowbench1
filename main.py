import os
import boto3
import json
import dotenv
# import openai
from openai import OpenAI
import PyPDF2
import io
from uuid import uuid4
import uuid
import pandas as pd

from streamlit_option_menu import option_menu

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.vectorstores import Pinecone
import pinecone

max_bytes = 40000
max_input_tokens_32k = 30000
max_input_tokens_64k = 60000
max_input_tokens_128k = 120000

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

import streamlit as st
# from streamlit_chat import message

from langchain.document_loaders import UnstructuredFileLoader

from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter 

from langchain.document_loaders import YoutubeLoader

from langchain.utilities import WikipediaAPIWrapper

from langchain.utilities import SerpAPIWrapper

from utils.sidebar import create_sidebar
from utils.initialize_session import initialize_session
from utils.clear_session import clear_session
from utils.trim_conversation_manage_mem import trim_conversation_history
from utils.get_num_tokens_from_string import get_num_tokens_from_string
from processors.process_text2image import process_text2image
from processors.process_huggingface import process_huggingface
from processors.process_wikipedia import process_wikipedia
from processors.process_openai import call_openai
from processors.process_image_infer import upload_to_s3_refactor, generate_presigned_url, get_inference



dotenv.load_dotenv(".env")
env_vars = dotenv.dotenv_values()
for key in env_vars:
    os.environ[key] = env_vars[key]


SERPAPI_API_KEY=os.getenv('SERPAPI_API_KEY')

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION')
    
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

PROMPT_INSERT_LAMBDA = os.getenv('PROMPT_INSERT_LAMBDA')
PROMPT_UPDATE_LAMBDA = os.getenv('PROMPT_UPDATE_LAMBDA')
PROMPT_QUERY_LAMBDA = os.getenv('PROMPT_QUERY_LAMBDA')
lambda_client = boto3.client('lambda', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ORGANIZATION = os.getenv('OPENAI_ORGANIZATION')

client = OpenAI(organization = OPENAI_ORGANIZATION, api_key = OPENAI_API_KEY)


if SERPAPI_API_KEY:
    serpapi_api_key = SERPAPI_API_KEY
else:
    print("SERPAPI_API_KEY key not found. Please make sure you have set the SERPAPI_API_KEY environment variable.")

pinecone.init (
    api_key = os.getenv('PINECONE_API_KEY'),
    environment = os.getenv('PINECONE_ENVIRONMENT')    
)


memory = ConversationBufferMemory(return_messages=True)
# Will come from AD
user_name_logged = "Rajesh Ghosh"
if 'curent_user' not in st.session_state:
    st.session_state['current_user'] = user_name_logged
if 'current_promptName' not in st.session_state:
        st.session_state['current_promptName'] = []


(
    model_name,
    persistence_choice,
    selected_sources,
    uploaded_files,
    summarize,
    youtube_url,
    k_similarity,
    kr_repos_chosen,
    task,
    upload_kr_docs_button,
    repo_selected_for_upload,
    kr_repos_list,
    embedding_model_name,
    selected_sources_image,
    macro_view,
    max_output_tokens,
    chunk_size,
    chunk_overlap,
    temperature_value,
    show_text_area,
    trigger_inference,
    domain_choice,
    privacy_setting
   
) = create_sidebar(st)
# print (f'macro_view right after sidebar call {macro_view}')


model=model_name

llm = OpenAI(model_name=model, temperature=temperature_value, max_tokens=max_output_tokens)

clear_button = None
#counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
if (task == 'Query'):
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
if 'entity_memory' not in st.session_state:
    st.session_state['entity_memory'] = ConversationEntityMemory (llm=llm, k=k_similarity)
Conversation = ConversationChain(llm=llm, prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, memory=st.session_state['entity_memory'] )

table_name = os.environ['PROCESS_TABLE']
# Initialise session state variables
initialize_session()

# ***************************************************

import tiktoken

def get_num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def trim_conversation_history(messages, max_messages=10, max_tokens=1000):
    print ("In trim_conversation_history")
    

    trimmed_messages = messages[-max_messages:]  # Keep only the last 'max_messages' messages

    total_tokens = 0
    start_index = 0
    for i, message in enumerate(reversed(trimmed_messages)):
        message_tokens = get_num_tokens_from_string(message['content'], 'cl100k_base')
        total_tokens += message_tokens
        if total_tokens > max_tokens:
            start_index = i
            break

    return trimmed_messages[-start_index:] if start_index > 0 else trimmed_messages

def get_gpt_response(client, model_name, messages, source, domain_choice):
    print ("In get_gpt_response")

    # Trim the conversation history to manage token and message limits
    messages = trim_conversation_history(messages)
    
    message_content, input_tokens, output_tokens, total_tokens, cost = call_openai(user_name_logged, user_input, model_name, messages, domain_choice)

    
        
    data = {
        "source": source,
        "response": message_content
                    
    }

    # Convert the dictionary to a JSON string
    try:
        json_data = json.dumps(data)
        
    except TypeError as e:
        print("TypeError occurred:", e)

    st.sidebar.write(f"**Usage Info:** ")
    st.sidebar.write(f"**Model:** {model_name}")
    st.sidebar.write(f"**Input Tokens:** {input_tokens}")
    st.sidebar.write(f"**Output Tokens:** {output_tokens}")
    st.sidebar.write(f"**Cost($):** {cost}")
    st.session_state['current_user'] = user_name_logged


    return json_data

# ***************************************************

    





# reset everything
if clear_button:
    print ("Clear button")
    clear_session()

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

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
def extract_distinct_file_paths(documents):
    file_paths = set()
    metadata_present = False

    for doc in documents:
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) and "file_path" in doc.metadata:
            file_paths.add(doc.metadata["file_path"])
            metadata_present = True

    return file_paths, metadata_present
def transform_format_a_to_b(format_a):
    documents = []
    for match in format_a['matches']:
        page_content = match['metadata']['text']
        metadata = {key: value for key, value in match['metadata'].items() if key != 'text'}
        documents.append(Document(page_content, metadata))
    return documents

def get_embedding(text, model="text-embedding-ada-002"):
   from openai import OpenAI
   client = OpenAI(organization = OPENAI_ORGANIZATION, api_key = OPENAI_API_KEY)
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


    

def search_vector_store (persistence_choice, index_name, user_input, model, source, k_similarity, kr_repos_chosen, domain_choice):
    print ('In search_vector_store', kr_repos_chosen)    
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
    print (result)
    dotenv.load_dotenv(".env")
    env_vars = dotenv.dotenv_values()
    for key in env_vars:
        os.environ[key] = env_vars[key]
    pinecone.init (
        api_key = os.getenv('PINECONE_API_KEY'),
        environment = os.getenv('PINECONE_ENVIRONMENT')   
    )
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')

    index_name = pinecone_index_name
    
    index = pinecone.Index(index_name)
    MODEL = "text-embedding-ada-002"

    prompt_embedding = get_embedding (result,model=MODEL)
    

    similarity_search_result = index.query(
        [prompt_embedding],
        top_k=k_similarity,
        filter={
          "repo": {"$in":kr_repos_chosen}
        },
        include_metadata=True
    )
    
    formatted_documents = transform_format_a_to_b(similarity_search_result)
    source_knowledge = "\n".join([x.page_content for x in formatted_documents])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {result}"""
    
    from langchain.schema import (
        SystemMessage,
        HumanMessage,
        AIMessage
    )
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
  
    ]
    
    userMessage = {"role": "user", "content": augmented_prompt}
       
    messages.append(userMessage)

    message_content, input_tokens, output_tokens, total_tokens, cost = call_openai(user_name_logged, user_input, model_name, messages, domain_choice)
    distinct_file_paths, metadata_present = extract_distinct_file_paths(formatted_documents)

    if metadata_present:
        data = {
            "source": source,
            "response": message_content,
            "distinct_file_paths" : list (distinct_file_paths ) ,
            "metadata_present" : True     
        }
    else:
        
        data = {
            "source": source,
            "response": message_content,
            "metadata_present": False
                       
        }
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)
    st.sidebar.write(f"**Usage Info:** ")
    st.sidebar.write(f"**Model:** {model_name}")
    st.sidebar.write(f"**Input Tokens:** {input_tokens}")
    st.sidebar.write(f"**Output Tokens:** {output_tokens}")
    st.sidebar.write(f"**Cost($):** {cost}")
    st.session_state['current_user'] = user_name_logged
    

    return json_data

def process_csv_file(file_path):
    print ("In process_csv_file")
    import pandas as pd
    from io import StringIO
    aws_bucket = os.getenv('S3_BUCKET_NAME')
    response = get_from_s3(aws_bucket, file_path)
  
    status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    
    if status_code == 200:
        # Read the CSV data from the S3 object
        csv_string = response['Body'].read().decode('utf-8')

        # Create a pandas DataFrame from the CSV string
        data_frame = pd.read_csv(StringIO(csv_string))

        chunk_size = 50  # This can be made configurable.
        chunks = []

        for start in range(0, len(data_frame), chunk_size):
            end = start + chunk_size
            chunk = data_frame[start:end]
            # chunks.append( 'page_content=   ' + "'" + chunk.to_string()+ "'")
            page_content = (
                f"page_content='{chunk.to_string(index=False)}\n'"
            )
            
            chunks.append(page_content)

        
        # return data_frame
    else:
        print(f"Error getting object {file_path} from bucket {aws_bucket}. Status code: {status_code}")
        return None
    if len(chunks) == 0:
        print ('No chunks extracted')
        return 0
    else:
        print (f'Number of chunks: {len(chunks)}')
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

def process_github(prompt, llm):
    from langchain.document_loaders import GitLoader

    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./example_data/test_repo2/",
        branch="master",
    )

    data = loader.load()
    

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
    prefix = """Have a conversation with a human, answering the following questions as best you can. Provide a detailed answer. You have access to the following tools:"""
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
   
    llm_chain = LLMChain(llm=OpenAI(temperature=temperature_value), prompt=promptAgent)
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

    source_mod = "Google"     
   
    data = {
        "source": source_mod,
        "response": response
    } 

    json_data = json.dumps(data)  
    return json_data


def process_YTLinks(youtube_video_url, user_input):
    print ('In process_YTLinks New', user_input)   
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

            # embeddings = OpenAIEmbeddings()        
            # docsearch = Pinecone.from_texts([t.page_content for t in chunks], embeddings, index_name=index_name)
            persistence_choice = "Pinecone"
            resp_YT = search_vector_store (persistence_choice, index_name, user_input, model, "YouTube", k_similarity,  kr_repos_chosen)
       
            data_dict = json.loads(resp_YT)
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
        
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def append_metadata(documents, file_path, repo_selected_for_upload):
    for doc in documents:
        doc.metadata["file_path"] = file_path
        doc.metadata["access"] = privacy_setting
        doc.metadata["repo"] = repo_selected_for_upload
        

def process_pdf_file(file_content, file_path, repo_selected_for_upload):
    print ('process_pdf_file')
    pdf_stream = io.BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    text_content = [page.extract_text() for page in pdf_reader.pages]
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.create_documents(text_content)  
    print (repo_selected_for_upload)
    append_metadata(chunks, file_path,repo_selected_for_upload)
   
    return chunks
    
def process_text_file_new(file_content, file_path, repo_selected_for_upload):
    print("In process_file_new")
    # print (file_content)
    text_content = file_content.decode('utf-8')     
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.create_documents([text_content]) 
    append_metadata(chunks, file_path, repo_selected_for_upload)  

    return chunks
    
def process_xlsx_file(s3,aws_bucket, file_path):
    print ("In process_xlsx_file")
    response = s3.get_object(Bucket=aws_bucket, Key=file_path)
    excel_data = response['Body'].read()
    from io import BytesIO
    import pandas as pd

# Use BytesIO to convert the byte stream to a file-like object
    excel_file = BytesIO(excel_data)

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(excel_file)

    from pandasai import SmartDataframe
    from pandasai import SmartDatalake
    from pandasai.llm import OpenAI
    llm = OpenAI(api_token=OPENAI_API_KEY)
    dl = SmartDatalake([df], config={"llm": llm})

      
    return df

def process_file(file_path, repo_selected_for_upload):
    print(f'Processing file: {file_path}')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    aws_bucket = os.getenv('S3_BUCKET_NAME')

    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)

    response = s3.get_object(Bucket=aws_bucket, Key=file_path)
    
    file_content = b""

    for chunk in response["Body"].iter_chunks():
        file_content += chunk
    # print (file_content)

    file_extension = os.path.splitext(file_path)[1][1:].lower()  # Get file extension without the dot

    if file_extension == 'pdf':
        chunks = process_pdf_file(file_content, file_path, repo_selected_for_upload)

    elif file_extension == 'txt':
        chunks = process_text_file_new(file_content, file_path, repo_selected_for_upload)
        
    elif file_extension == 'csv':
        chunks = process_csv_file(s3,aws_bucket, file_path)
        
    elif file_extension == 'xlsx':
        chunks = process_xlsx_file(s3,aws_bucket, file_path)

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
   
    return chunks

def upload_to_s3(bucket_name, uploaded_file):
    bytes_data = uploaded_file.read()
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    aws_bucket = bucket_name
    aws_bucket_input_path = os.getenv('S3_BUCKET_INPUT_PATH')
    # Create an S3 client
    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)

    # Define the target path in S3
    s3_target_path = aws_bucket_input_path + uploaded_file.name

    s3.put_object(Body=bytes_data, Bucket=aws_bucket, Key=s3_target_path)
    return s3_target_path

def get_from_s3(bucket_name, path_name):

    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION')

    # Create an S3 client
    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
    response = s3.get_object(Bucket=bucket_name, Key=path_name)
    return response
    
def extract_chunks_from_uploaded_file(uploaded_file, repo_selected_for_upload):
    print('In extract_chunks_from_uploaded_file')
    bucket_name = os.getenv('S3_BUCKET_NAME')
    s3_target_path = upload_to_s3(bucket_name,uploaded_file)
    # _ is used to ignore the base name of the file
    _, file_extension = os.path.splitext(uploaded_file.name)
        

    if file_extension.lower() == '.pdf': 
        chunks = process_file(s3_target_path, repo_selected_for_upload)
        print ("pdf_chunks: ", len(chunks))
    elif file_extension.lower() == '.txt':
        print ("Processing .txt ")        
        chunks = process_file(s3_target_path, repo_selected_for_upload)
    elif file_extension.lower() == '.csv':
        chunks = process_csv_file(s3_target_path)
    elif file_extension.lower() == '.docx':
        chunks = process_docx_file(uploaded_file.name)
    elif file_extension.lower() == '.xlsx' or file_extension.lower() == '.xls':
        chunks = process_file(s3_target_path)
    elif file_extension.lower() == '.pptx':
        chunks = process_pptx_file(uploaded_file.name)
    else:
        raise ValueError(f'Unsupported Filetype: {file_extension}')

    return chunks

def process_openai(prompt, model, Conversation, kr_repos_chosen, domain_choice):
    print ("In process_openai ")


    # Initialize or retrieve session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    json_data = get_gpt_response(client, model_name, st.session_state.messages, "Open AI", domain_choice)
    return json_data
    


def process_knowledge_base(prompt, model, Conversation, kr_repos_chosen, domain_choice):
    print ('In process_knowledge_base ')    
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
    # Need to refactor Rajesh
    embed = OpenAIEmbeddings(
        model=embedding_model_name,
        openai_api_key=OPENAI_API_KEY
    )
    index_name = pinecone_index_name
    # pinecone_index = pinecone.Index(index_name)

 
    # Rajesh change
    resp = search_vector_store (persistence_choice, index_name, prompt, model, "KR", k_similarity, kr_repos_chosen, domain_choice)

    return resp
    
def process_uploaded_file(uploaded_files,  persistence_choice, repo_selected_for_upload):
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
            chunks = extract_chunks_from_uploaded_file(uploaded_file, repo_selected_for_upload)

            if chunks == 0:
                print("No chunks extracted from: ", uploaded_file)                    
            else:    
                print(f'Processing file {index + 1}: {uploaded_file.name}')
                st.write(f'Processing file {index + 1}: {uploaded_file.name}')            
                docs_chunks.extend(chunks) 
                print (f'Total number of chunks : {len(docs_chunks)}')


        if repo_selected_for_upload in kr_repos_list:
            print(f'processing {repo_selected_for_upload} KR')

            print (f'Number of chunks in docs_chunks {len(docs_chunks)}')
            dotenv.load_dotenv(".env")
            env_vars = dotenv.dotenv_values()
            for key in env_vars:
                os.environ[key] = env_vars[key]
                
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            # if OPENAI_API_KEY:
            #     openai.api_key = OPENAI_API_KEY
            pinecone.init (
                api_key = os.getenv('PINECONE_API_KEY'),
                environment = os.getenv('PINECONE_ENVIRONMENT')   
            )
            pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
            client = OpenAI(organization = OPENAI_ORGANIZATION, api_key = OPENAI_API_KEY)
            embeddings = OpenAIEmbeddings()
            try:
                index_name = pinecone_index_name  # Specify the index name as a string
                Pinecone.from_documents(
                    docs_chunks, embeddings, index_name=index_name
                )

            except pinecone.exceptions.PineconeException as e:
                print(f"An error occurred: {str(e)}")


            print ("Vector Store Loaded!")
            return repo_selected_for_upload

def selected_data_sources(selected_elements, prompt, model, llm, Conversation, kr_repos_chosen, kr_repos_list, domain_choice):
    print ("In selected_data_sources")
    import json

    all_responses = []
    selected_elements_functions = {
       
        'Wikipedia': process_wikipedia,       
        'KR':process_knowledge_base,
        'Open AI': process_openai,
        'Google':process_google_search,       
        'Hugging Face':process_huggingface,
        'YouTube': process_YTLinks,
        
    }

    for element in selected_elements:
        if element in selected_elements_functions:

            if (element == 'Open AI'):
                str_response = selected_elements_functions[element](prompt, model, Conversation, domain_choice)
                json_response = json.loads(str_response)
                
               
                all_responses.append(json_response)
                
            elif (element == 'KR'):
                print ('Processing KR')
                str_response = selected_elements_functions[element](prompt, model, Conversation, kr_repos_chosen, domain_choice)
                print (str_response)
                json_response = json.loads(str_response)
                print (json_response)
                all_responses.append(json_response)
                
            elif (element == 'Google'):
                print ('Processing Google')
                str_response = selected_elements_functions[element](prompt, llm)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
                
            elif (element == 'YouTube'):
                print ('Processing YouTube')
                str_response = selected_elements_functions[element](youtube_url , prompt)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
                
            elif (element == 'Hugging Face'):
                str_response = selected_elements_functions[element](prompt,llm)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
                
            elif (element == 'Wikipedia'):
                str_response = selected_elements_functions[element](prompt, llm)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
            else:
                print ("check chosen sources")
            accumulated_json = {"all_responses": all_responses}
            print (accumulated_json)

      

            accumulated_json_str = json.dumps(accumulated_json)


    return accumulated_json_str

def update_prompt(like_status, comments):
    print ("In update_prompt ", st.session_state['current_promptName'])

    data = {
        "userName": st.session_state['current_user'],
        "promptName": st.session_state['current_promptName'],
        "feedback": like_status,
        "comments": comments
    }
    print (data)

    lambda_function_name = PROMPT_UPDATE_LAMBDA
    lambda_response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(data)
    )

    if lambda_response['StatusCode'] != 200:
        raise Exception(f"AWS Lambda invocation failed with status code: {lambda_response['StatusCode']}")

    # Function to format date
def format_date(date_str):
    from datetime import datetime
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    if date_obj.date() == datetime.today().date():
        return 'Today'
    elif date_obj.date() == (datetime.today() - pd.Timedelta(days=1)).date():
        return 'Yesterday'
    else:
        return date_obj.strftime('%d%b%y')


def get_response(user_input, kr_repos_chosen):
    import json
    print ('In get_response...', kr_repos_chosen)
    
    if 'text2Image' in selected_sources_image and user_input and goButton:
        with st.spinner("Generating image..."):
            with response_container:
                 process_text2image(user_input)
    else:  

        if  user_input and len(selected_sources) > 0 and goButton:

          with st.spinner("Searching requested sources..."):        
            str_resp = selected_data_sources(selected_sources, user_input, model_name, llm, Conversation, kr_repos_chosen, kr_repos_list, domain_choice)               
            data = json.loads(str_resp)['all_responses']

            response_dict = {
                'wiki_response': None,
                'google_response': None,
                'openai_response': None,
                'hugging_face_response': None,
                'youtube_response':None,
                'kr_response':None
            }

            if not trigger_inference:
                for response in data:
                    source = response['source']
                    if source in selected_sources:
                        
                        if source == "KR":
                            if (response['metadata_present']):
                                response_dict[f"{source.lower().replace(' ', '_')}_response"] = response['response'] + "\n\n" + "Doc Sources: " + ", ".join(response['distinct_file_paths'])
                            else:
                                response_dict[f"{source.lower().replace(' ', '_')}_response"] = response['response'] + "\n\n" + "Doc Sources: "  + "NA"   
                        else:                        
                            response_dict[f"{source.lower().replace(' ', '_')}_response"] = response['response']
           
            st.session_state["user_prompts"].append(user_input)        
            st.session_state['sel_source'].append(selected_sources)
            
            wiki_response = response_dict.get('wikipedia_response')
            kr_response = response_dict.get('kr_response')            
            google_response = response_dict.get('google_response')
            openai_response = response_dict.get('open_ai_response')
            huggingface_response = response_dict.get('hugging_face_response')
            youtube_response = response_dict.get('youtube_response')
            
            all_response_str = ''

            if wiki_response:
                print ('in wiki response')
                st.session_state['generated_wiki'].append(wiki_response)
                all_response_str = all_response_str + "From Wikipedia  \n" + "-----------------------------" + "\n\n" + wiki_response + "\n\n"

            if youtube_response:
                print ('in YouTube response')
                st.session_state['generated_youtube'].append(youtube_response)
                all_response_str = all_response_str + "From YouTube  \n" + "-----------------------------" + "\n\n" + youtube_response + "\n\n"
            
            if google_response:
                print ('in google response')
                st.session_state['generated_google'].append(google_response)
                all_response_str = all_response_str + "From Google  \n" + "-----------------------------" + "\n\n" + google_response + "\n\n"

            if openai_response:
                print ('in openai response')
                st.session_state['generated_openai'].append(openai_response)
                all_response_str = all_response_str + "From OpenAI  \n" + "----------------------" + "\n\n" + openai_response + "\n\n"
                
            if kr_response:
                print ('kr_response')            
                st.session_state['generated_KR'].append(kr_response)
                choice_str = ', '.join(kr_repos_chosen) if kr_repos_chosen else "None selected"
                # all_response_str = all_response_str + "<h4>From KR: " + choice_str + "</h4>" + "<hr>" + "<p>" + kr_response + "</p>"
                all_response_str = all_response_str + "From KR: " + choice_str +  "\n" + "----------------------" + "\n\n" + kr_response + "\n\n"
                
            if huggingface_response:
                st.session_state['generated_hf'].append(huggingface_response)
                all_response_str = all_response_str + "From Hugging Face  \n" + "-----------------------------" + "\n\n" + huggingface_response + "\n\n"
              
            else:
                print("Uploads data is not available.")
                


           
            st.session_state["all_response_dict"].append (all_response_str)
            st.session_state['generated_response'].append(all_response_str)

            if st.session_state['all_response_dict']:
                with response_container:

                    download_str = []

                    latest_index = len(st.session_state['generated_response']) - 1

                    st.info(st.session_state["user_prompts"][latest_index], icon="✅")
                    st.success(st.session_state["generated_response"][latest_index], icon="✅")
                    download_str.append(st.session_state["user_prompts"][latest_index])
                    download_str.append(st.session_state["generated_response"][latest_index])
                                   
                    if summarize:
                        summary_dict = []
                        st.subheader('Summary from all sources')
                        generated_string = str(st.session_state['generated_response'][-1])

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
            uploaded_kr = process_uploaded_file(uploaded_files,  persistence_choice, repo_selected_for_upload)
            st.write ("Done!")
             
    if (task =='Query'):        
        
        if 'text2Image' in selected_sources_image:
            selected_sources_text = 'text2Image'
            default_text_value = "Generate a high-resolution image with the exceptional sharpness, clarity, and color fidelity characteristic of a professional photograph taken with a Canon EOS-1D C DSLR Camera. Ensure the image exhibits impeccable details, vibrant colors, and minimal noise, emulating the superior image quality of this renowned camera. Capture the essence of professional photography, with a focus on realistic textures, crisp edges, and accurate representation of lighting. The final output should resemble a masterpiece produced by the Canon EOS-1D C"
            placeholder_default = None
            ask_text = "**Selected Sources:** " + "**:green[" + selected_sources_text + "]**" 
            user_input = st.text_area(ask_text, height=150, key='input', value = default_text_value, placeholder=placeholder_default, label_visibility="visible") 
            
        else:
            if macro_view:
                try:
                    dotenv.load_dotenv(".env")
                    env_vars = dotenv.dotenv_values()
                    for key in env_vars:
                        os.environ[key] = env_vars[key]
                    pinecone.init (
                        api_key = os.getenv('PINECONE_API_KEY'),
                        environment = os.getenv('PINECONE_ENVIRONMENT')   
                    )
                    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
                    index_name = pinecone_index_name  # Specify the index name as a string
                    indexPinecone = pinecone.Index(index_name)
                    num_vectors = indexPinecone.describe_index_stats()
                    num_vectors_cnt = num_vectors.namespaces[''].vector_count
                    st.info (f'Macro View selected, set k to {num_vectors_cnt} and choose model with the max context length in settings ' )
                  
                except pinecone.exceptions.PineconeException as e:
                    print(f"An error occurred: {str(e)}")
            trigger_inference_image_uploaded = False
                
            if trigger_inference:  
                print ("In trigger_inference")  
                selected_sources_text = "Image Inference"
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])  
                if uploaded_file is not None:
                    st.write("")
                    st.write("")
                    print ("Uploaded File not none ", uploaded_file.name)
                    trigger_inference_image_uploaded = True
                    st.image(uploaded_file, caption='Uploaded Image: ' + uploaded_file.name, use_column_width=True)
                    placeholder_default = "Ask something about the image..."
                    if show_text_area:
                        user_input = st.text_area("Ask:", placeholder=placeholder_default, height=300)
                    else:
                        user_input = st.text_input("Ask:",  key='input', placeholder=placeholder_default, label_visibility="visible") 

                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    with col1:
                        goButton = st.button("Go")     
                    with col2:
                        add_to_library = st.button("Add to 📚", help = "add this prompt to your prompt library")               
                    with col3:
                        show_library = st.button("Your 📚", help = "click to see your prompt library") 
                    with col4:
                        improve_button = st.button("Improve", type="primary", help = "report this prompt to for investigation")    
                    if user_input:                      

                        if goButton:
                            s3_path = upload_to_s3_refactor("bucket-company-static-assets", uploaded_file)
                            image_url = generate_presigned_url("bucket-company-static-assets", s3_path)
                            response = get_inference(user_input, image_url)
                            st.write(response)
                else:
                    trigger_inference_image_uploaded = False
            else:  
                selected_sources_text = ", ".join(selected_sources)                
                default_text_value = ''
                placeholder_default = "Ask ..."
                ask_text = "**Selected Sources:** " + "**:green[" + selected_sources_text + "]**" 
                if show_text_area:
                    # Text area for longer prompts
                    user_input = st.text_area(ask_text, placeholder=placeholder_default, height=300)
                else:
                    user_input = st.text_input(ask_text,  key='input', value = default_text_value, placeholder=placeholder_default, label_visibility="visible") 
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    goButton = st.button("Go")   
                                    
                with col3:
                    show_library = st.button("Your 📚", help = "click to see your prompt library")
                with col2:
                    add_to_library = st.button("Add to 📚", help = "add this prompt to your prompt library") 
                with col4:
                    improve_button = st.button("Improve", type="primary", help = "report this prompt to for investigation")         
                
                if goButton:         
                    get_response (user_input, kr_repos_chosen)
                    
                # Display the conversation history using an expander, and allow the user to download it.
        words_in_input_text = False
        if ((trigger_inference_image_uploaded and trigger_inference) or (not trigger_inference)):
           
            if user_input.strip():  # This ensures that empty spaces are not counted
                words = user_input.split()
                if len(words) >= 1:
                    words_in_input_text = True

            if words_in_input_text:
                if add_to_library:
                    st.session_state.button_pressed = "Add to Library"
                    st.session_state.feedback_given = False 
                if improve_button:
                    st.session_state.button_pressed = "Improve"
                    st.session_state.feedback_given = False 

            if st.session_state.button_pressed and not st.session_state.feedback_given:
                placeholder = st.empty()
                with placeholder.form(key='comment_form'):
                    comments = st.text_input("Enter comments:")
                    submit_button = st.form_submit_button("Submit")

            # If the submit button is pressed, handle the submission and clear the placeholder
                if submit_button:
                    update_prompt(st.session_state.button_pressed, comments)
                    st.info("Noted")
                    st.session_state.feedback_given = True
                    st.session_state.button_pressed= False
                    placeholder.empty()  # This clears the form


        if not trigger_inference_image_uploaded and  trigger_inference:
            show_library = st.button("Your 📚", help = "click to see your prompt library")

        if show_library:
                add_to_library_str = "Add to Library"              
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
                items = response_payload.get("Items", [])

                if items:
                    st.markdown("Curated prompts saved by <b>{}</b>".format(st.session_state['current_user']), unsafe_allow_html=True)

                    # Create a Pandas DataFrame with necessary fields including 'datetimestamp' and 'domain'
                    df = pd.DataFrame([
                        {"Prompt": item["prompt"]["S"], "Comments": item["comments"]["S"],
                        "Date": item["date"]["S"], "Time": item["time"]["S"], "DateTimestamp": item["datetimestamp"]["S"], 
                        "domain": item.get("domain", {}).get("S", "Not Specified")}  # Provide a default value if 'domain' is not present
                        for item in items
                    ])  

                    # Convert 'DateTimestamp' to datetime and sort in descending order
                    df['DateTimestamp'] = pd.to_datetime(df['DateTimestamp'])
                    df.sort_values(by='DateTimestamp', ascending=False, inplace=True)

                    # Apply formatting to the 'Date' and 'Time' columns
                    df['Date'] = df['Date'].apply(format_date)
                    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.strftime('%I:%M %p').str.lower()

                    # Filter the DataFrame based on the domain choice
                    filtered_df = df[df['domain'].astype(str) == domain_choice]

                    # Define columns to display: 'Prompt', 'Comments', 'Date', 'Time', and possibly 'domain'
                    df_display = filtered_df[['Prompt', 'Comments', 'Date', 'Time']]                    

                    # Display the filtered DataFrame with Streamlit, setting the counter to start from 1
                    st.table(df_display.assign(Counter=range(1, len(df_display) + 1)).set_index('Counter'))
                else:
                    st.write("No items found with 'Add to Library' feedback.")

                    
        with st.expander("Download Conversation", expanded=False):
            download_str = []
           
            for i in range(len(st.session_state['generated_response'])-1, -1, -1):
                st.info(st.session_state["user_prompts"][i],icon="✅")
                st.success(st.session_state["generated_response"][i], icon="✅")
                download_str.append(st.session_state["user_prompts"][i])
                download_str.append(st.session_state["generated_response"][i])

            download_str = '\n'.join(download_str)
            if download_str:
                st.download_button('Download',download_str)
                    
                    