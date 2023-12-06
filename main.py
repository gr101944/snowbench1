import os
import boto3
import json
import dotenv
import openai
import PyPDF2
import io
from uuid import uuid4
import uuid


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
from streamlit_chat import message

from langchain.document_loaders import UnstructuredFileLoader

from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter 

from langchain.document_loaders import YoutubeLoader

from langchain.utilities import WikipediaAPIWrapper
from langchain import HuggingFaceHub
from langchain.utilities import SerpAPIWrapper

from utils.sidebar import create_sidebar
from utils.initialize_session import initialize_session
from utils.clear_session import clear_session
from processors.process_text2image import process_text2image

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
user_name_logged = "Rajesh Ghosh"
if 'curent_user' not in st.session_state:
    st.session_state['current_user'] = user_name_logged
if 'curent_promptName' not in st.session_state:
        st.session_state['curent_promptName'] = []


(
    model_name,
    persistence_choice,
    selected_sources,
    uploaded_files,
    summarize,
    youtube_url,
    k_similarity,
    sources_chosen,
    task,
    upload_kr_docs_button,
    ingest_source_chosen,
    source_data_list,
    source_category,
    embedding_model_name,
    selected_sources_image,
    macro_view,
    max_output_tokens,
    chunk_size,
    chunk_overlap,
    temperature_value
) = create_sidebar(st)
print (f'macro_view right after sidebar call {macro_view}')

model=model_name
print (model)
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



    
def process_Wikipedia(prompt, llm):
    print("Processing Wikipedia...", prompt)
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
    
def count_bytes_in_string(input_string):
    try:
        byte_count = len(input_string.encode('utf-8'))
        print(f'The number of bytes in the string is: {byte_count}')
        return byte_count 
        
    except Exception as e:
        print(f"An error occurred: {e}")
    

def search_vector_store3 (persistence_choice, VectorStore, user_input, model, source, k_similarity, promptId_random):
    print ('In search_vector_store3', model)
    print ("k_similarity ", k_similarity)
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
        
        
        
        data = {    
            "userName": user_name_logged,
            "promptName": promptId_random,
            "prompt": user_input,
            "completion": response['output_text'],
            "summary": "No Summary",
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cost": cost,
            "feedback": ""
        }

        st.session_state['current_user'] = user_name_logged
        st.session_state['curent_promptName'] = promptId_random

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
            print ("Success calling lambda!")
        st.sidebar.write(f"**Usage Info:** ")
        st.sidebar.write(f"**Model:** {model_name}")
        st.sidebar.write(f"**Input Tokens:** {input_tokens}")
        st.sidebar.write(f"**Output Tokens:** {output_tokens}")
        st.sidebar.write(f"**Cost($):** {cost}")
        # st.write(f"**Input Tokens:** {input_tokens} | **Output Tokens:** {output_tokens} | **Cost:** {cost}")


    if "I don't know" in response or response == None:       
        st.write('Info not in artefact...') 
   

    data = {
            "source": source,
            "response": response
    }
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)

    return json_data

def process_csv_file(s3,aws_bucket, file_path):
    print ("In process_csv_file")
    import pandas as pd
    from io import StringIO
    response = s3.get_object(Bucket=aws_bucket, Key=file_path)
    status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    
    if status_code == 200:
        # Read the CSV data from the S3 object
        csv_string = response['Body'].read().decode('utf-8')

        # Create a pandas DataFrame from the CSV string
        data_frame = pd.read_csv(StringIO(csv_string))
        print ("*********************")
        print (len(data_frame))
        chunk_size = 50  # This can be made configurable.
        chunks = []

        for start in range(0, len(data_frame), chunk_size):
            end = start + chunk_size
            chunk = data_frame[start:end]
            # chunks.append( 'page_content=   ' + "'" + chunk.to_string()+ "'")
            page_content = (
                f"page_content='{chunk.to_string(index=False)}\n'"
            )
            print (page_content)
            chunks.append(page_content)
            # chunks.append({"page_content": chunk.to_string()})
            # print (chunks)
            # chunks.append(chunk.to_string()) 
            # print (data_frame)
        
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
    print (len(data))

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

            embeddings = OpenAIEmbeddings()        
            docsearch = Pinecone.from_texts([t.page_content for t in chunks], embeddings, index_name=index_name)
            persistence_choice = "Pinecone"
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



def process_hugging_face(question):
    print ('In process_hugging_face')
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
        repo_id=repo_id, model_kwargs={"temperature": temperature_value, "max_new_tokens": max_output_tokens}
    )
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


def process_pdf_file(file_content):
    print ('process_pdf_file')
    pdf_stream = io.BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    text_content = [page.extract_text() for page in pdf_reader.pages]
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.create_documents(text_content)  
    return chunks
    
def process_text_file_new(file_content):
    print("In process_file_new")
    text_content = file_content.decode('utf-8')     
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.create_documents([text_content])   
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

    # Now df contains your data as a pandas DataFrame
    print(df)
    from pandasai import SmartDataframe
    from pandasai import SmartDatalake
    from pandasai.llm import OpenAI
    llm = OpenAI(api_token=OPENAI_API_KEY)
    dl = SmartDatalake([df], config={"llm": llm})
    print ("**********************")
    print (dl.chat("how many rows are there"))
    print (dl.chat("what are the action items for 2021"))
    print ("**********************")
    
    # document_instance = xlsx_content[0]
    # page_content = document_instance.page_content

    # print(page_content)

 
    # # text_content = file_content.decode('utf-8')   
    # print (chunk_size) 
    # print (chunk_overlap) 
    # text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    # chunks = text_splitter.create_documents(page_content)        
    return df

def process_file(file_path):
    print(f'Rajesh New function Processing file: {file_path}')
    from io import StringIO
    import pandas as pd

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
        chunks = process_pdf_file(file_content)

    elif file_extension == 'txt':
        chunks = process_text_file_new(file_content)
        
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

    s3.put_object(Body=bytes_data, Bucket=aws_bucket, Key=s3_target_path)
    

    # _ is used to ignore the base name of the file
    _, file_extension = os.path.splitext(uploaded_file.name)
        

    if file_extension.lower() == '.pdf': 
        chunks = process_file(s3_target_path)
        print ("pdf_chunks: ", len(chunks))
    elif file_extension.lower() == '.txt':
        print ("Processing .txt ")        
        chunks = process_file(s3_target_path)
    elif file_extension.lower() == '.csv':
        chunks = process_csv_file(s3,aws_bucket, s3_target_path)
    elif file_extension.lower() == '.docx':
        chunks = process_docx_file(uploaded_file.name)
    elif file_extension.lower() == '.xlsx' or file_extension.lower() == '.xls':
        chunks = process_file(s3_target_path)
    elif file_extension.lower() == '.pptx':
        chunks = process_pptx_file(uploaded_file.name)
    else:
        raise ValueError(f'Unsupported Filetype: {file_extension}')

    return chunks

def process_openai(prompt, model, Conversation):
    print ("In process_openai")
    st.session_state['messages'].append({"role": "user", "content": prompt})
    print (prompt)
    response = Conversation.run(input=prompt)
    print (response)
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

def process_knowledge_base(prompt, model, Conversation, sources_chosen, source_data_list, promptId_random):
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
    # Rajesh change
    resp = search_vector_store3 (persistence_choice, vectorstore, prompt, model, "KR", k_similarity, promptId_random)

    return resp

def text_to_vector(text):
    # Replace this with the actual code to convert text to vectors using your embedding method
    # This is a placeholder function
    # Ensure the vectors have the correct dimension (1536 in this case)
    return [0.0] * 1536  # Placeholder for a vector of dimension 1536
    
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
                
            # metadata_list = [{"Auth": "Exec"}, {"Access": "Restricted"}] 
            # print ("Example chunk")
            # print (docs_chunks[0])
            # docs_chunks_with_metadata = list(zip(docs_chunks, metadata_list))
            # indexPinecone.upsert(docs_chunks_with_metadata)
            # print ("Pinecone updated")
            # docs_chunks_with_metadata = [{"id": str(i), "values": text_to_vector(doc.page_content), "metadata": metadata} for i, (doc, metadata) in enumerate(zip(docs_chunks, metadata_list))]
            # docs_chunks_with_metadata = [{"id": str(i), "values": [float(value) for value in doc.page_content.split()], "metadata": metadata} for i, (doc, metadata) in enumerate(zip(docs_chunks, metadata_list))]
            # docs_chunks_with_metadata = list(zip(docs_chunks, metadata_list))
            # print ("done 1")
            # embed_and_upsert_chunks(docs_chunks, index)
            # print ("upsert done")
            # docs_chunks_with_metadata = list(zip([t.page_content for t in docs_chunks], metadata_list))
            # indexPinecone.upsert(docs_chunks_with_metadata)
            docsearch = Pinecone.from_texts([t.page_content for t in docs_chunks], embeddings, index_name=index_name)
            num_vectors = indexPinecone.describe_index_stats()
            num_vectors_cnt = num_vectors.namespaces[''].vector_count
            print (num_vectors)
            print (num_vectors_cnt)
            return ingest_source_chosen

def selected_data_sources(selected_elements, prompt, model, llm, Conversation, sources_chosen, source_data_list, promptId_random):
    print ("In selected_data_sources")
    import json

    all_responses = []
    selected_elements_functions = {
       
        'Wikipedia': process_Wikipedia,       
        'KR':process_knowledge_base,
        'Open AI': process_openai,
        'Google':process_google_search,       
        'Hugging Face':process_hugging_face,
        'YouTube': process_YTLinks,
        
    }

    for element in selected_elements:
        if element in selected_elements_functions:

            if (element == 'Open AI'):
                str_response = selected_elements_functions[element](prompt, model, Conversation)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
                
            elif (element == 'KR'):
                print ('Processing KR')
                str_response = selected_elements_functions[element](prompt, model, Conversation, sources_chosen, source_data_list, promptId_random)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
                
            elif (element == 'Google'):
                print ('Processing Google')
                str_response = selected_elements_functions[element](prompt, llm, promptId_random)
                json_response = json.loads(str_response)
                all_responses.append(json_response)
                
            elif (element == 'YouTube'):
                print ('Processing YouTube')
                str_response = selected_elements_functions[element](youtube_url , prompt, promptId_random)
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

def update_prompt(like_status, comments):
    print ("In update_prompt")
    data = {
        "userName": st.session_state['current_user'],
        "promptName": st.session_state['curent_promptName'],
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

def get_response(user_input, source_data_list, promptId_random):
    import json
    print ('In get_response...')
    
    if 'text2Image' in selected_sources_image and user_input and goButton:
        with st.spinner("Generating image..."):
            with response_container:
                 process_text2image(user_input, promptId_random)
    else:  

        if  user_input and len(selected_sources) > 0 and goButton:
          print ("Go pressed")

          with st.spinner("Searching requested sources..."):        
            str_resp = selected_data_sources(selected_sources, user_input, model_name, llm, Conversation, sources_chosen, source_data_list, promptId_random)               
            data = json.loads(str_resp)['all_responses']

            response_dict = {
                'wiki_response': None,
                'google_response': None,
                'openai_response': None,
                'hugging_face_response': None,
                'youtube_response':None,
                'kr_response':None
            }
            for response in data:
                source = response['source']
                if source in selected_sources:
                  
                    response_dict[f"{source.lower().replace(' ', '_')}_response"] = response['response']
           
            st.session_state["past"].append(user_input)
        
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
                choice_str = ', '.join(sources_chosen) if sources_chosen else "None selected"
                all_response_str = all_response_str + "From KR: " + choice_str +  "\n" + "----------------------" + "\n\n" + kr_response['output_text'] + "\n\n"
                
            if huggingface_response:
                st.session_state['generated_hf'].append(huggingface_response)
                all_response_str = all_response_str + "From Hugging Face  \n" + "-----------------------------" + "\n\n" + huggingface_response + "\n\n"
              
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
                    # Rajesh to come
                
            selected_sources_text = ", ".join(selected_sources)
            default_text_value = ''
            placeholder_default = "Ask something..."
            ask_text = "**Selected Sources:** " + "**:green[" + selected_sources_text + "]**" 
            user_input = st.text_input(ask_text,  key='input', value = default_text_value, placeholder=placeholder_default, label_visibility="visible") 
        
  

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            goButton = st.button("Go")

        with col3:
            add_to_library = st.button("Add to Library")       

        with col4:
            improve_button = st.button("Improve", type="primary")         
             
        if add_to_library:
            st.session_state.button_pressed = "Add to Library"
            st.session_state.feedback_given = False 
        if improve_button:
            st.session_state.button_pressed = "Improve"
            st.session_state.feedback_given = False 


        # Check if any button is pressed and feedback is not given
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

    
        if goButton:
            random_string = str(uuid.uuid4())
            promptId_random = "prompt-" + random_string
            get_response (user_input, source_data_list, promptId_random)
            download_str = []
        # Display the conversation history using an expander, and allow the user to download it.
            with st.expander("Download Conversation", expanded=False):
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    st.info(st.session_state["past"][i],icon="✅")
                    st.success(st.session_state["generated"][i], icon="✅")
                    download_str.append(st.session_state["past"][i])
                    download_str.append(st.session_state["generated"][i])
 
                download_str = '\n'.join(download_str)
                if download_str:
                    st.download_button('Download',download_str)
        if st.button("Your Prompt Library"):
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
            import pandas as pd
            
            
            if items:
                st.markdown("Curated prompts saved by <b>{}</b>".format(st.session_state['current_user']), unsafe_allow_html=True)
                    # Create a Pandas DataFrame
                df = pd.DataFrame([
                    {"Prompt": item["prompt"]["S"], "Comments": item["comments"]["S"], "PromptName": item["promptName"]["S"]}
                    for item in items
                ])


                # Display the DataFrame with Streamlit, setting the counter to start from 1
                st.table(df.assign(Counter=range(1, len(df) + 1)).set_index('Counter'))
            else:
                    st.write("No items found with 'Add to Library' feedback.")
         
           