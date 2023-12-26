import os
import dotenv
import re

dotenv.load_dotenv(".env")
env_vars = dotenv.dotenv_values()
for key in env_vars:
    os.environ[key] = env_vars[key]
STATIC_ASSEST_BUCKET_URL = os.getenv("STATIC_ASSEST_BUCKET_URL")
STATIC_ASSEST_BUCKET_FOLDER = os.getenv("STATIC_ASSEST_BUCKET_FOLDER")
UPDATE_CONFIG_LAMBDA = os.getenv('UPDATE_CONFIG_LAMBDA')
QUERY_CONFIG_LAMBDA = os.getenv ('QUERY_CONFIG_LAMBDA')
LOGO_NAME = os.getenv("LOGO_NAME")  
import pinecone

import streamlit as st
import pinecone


def is_valid_repo_name(name):

    return bool(re.match("^[A-Za-z0-9_\- ]*$", name))

def delete_repository_docs(repo_name):
    try:
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
        index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME'))
        index.delete(
            filter={
                "repo": {"$eq": repo_name} 
            }
        )
        st.sidebar.success(f"Successfully deleted documents from {repo_name}.")
    except Exception as e:
        st.sidebar.error(f"Failed to delete documents from {repo_name}: {e}")

def invoke_lambda_function(function_name, payload):
    import boto3
    import json
    lambda_client = boto3.client('lambda')
    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        response_payload = json.load(response['Payload'])
        return response_payload
    except Exception as e:
        st.error(f"Error invoking Lambda function: {e}")
        return None


def update_repository_list(repo_list):
    # Update the repository list using updateConfig Lambda function
    invoke_lambda_function(UPDATE_CONFIG_LAMBDA, {'configName': 'Banking', 'configValue': repo_list})
    
def get_repository_list(domain_choice):
    import json
    response = invoke_lambda_function('queryConfig', {'configName': domain_choice})
    if response and 'statusCode' in response and response['statusCode'] == 200:
        # Parse the JSON string in the 'body' key
        body = json.loads(response['body'])
        if body and isinstance(body, list) and 'configValue' in body[0]:
            return body[0]['configValue']
    return []

  

def create_sidebar (st):


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
    html_code = """
        <style>
            div.row-widget.stRadio > div{flex-direction:row;}
        </style>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    
    image_path = os.path.join(STATIC_ASSEST_BUCKET_URL, STATIC_ASSEST_BUCKET_FOLDER, LOGO_NAME)
    
    with st.sidebar:
        st.image(image_path, width=55)
    

    
    task_list = [  'Query', 'Data Load']
    
    other_sources = ['Open AI', 'YouTube', 'Google', 'KR', 'text2Image']
    text2Image_source = ['text2Image']
    other_sources_default = ['KR']
    embedding_options = ['text-embedding-ada-002']
    persistence_options = [ 'Pinecone']
    model_options = ['gpt-3.5-turbo','gpt-4', 'gpt-4-1106-preview','gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k']
    # 
    url = "https://www.persistent.com/"
    
    with st.sidebar.expander("Select Your Domain"):
    # Radio button with options
        domain_choice = st.radio(
            "Choose a sector:",
            ('Banking', 'Healthcare', 'CMT')  # Options for the radio button
        )
    kr_repos_list = get_repository_list(domain_choice)

    kr_repos_list_default = kr_repos_list[0]
    if 'kr_repos_list' not in st.session_state:
        st.session_state['kr_repos_list'] = kr_repos_list
    else:
        st.session_state['kr_repos_list'] = kr_repos_list
    with st.sidebar.expander(" 🛠️ LLM Configurations ", expanded=False):
    # Option to preview memory store
        show_text_area = st.checkbox("Show text area for longer prompts")
        model_name  = st.selectbox(label='LLM:', options=model_options, help='GPT-4 in waiting list 🤨')
        embedding_model_name  = st.radio('Embedding:', options=embedding_options, help='Option to change embedding model, keep in mind to match with the LLM 🤨')
        persistence_choice = st.radio('Persistence', persistence_options, help = "Using Pinecone...")
        chunk_size = st.number_input ("Chunk Size",value= 1000)
        chunk_overlap = st.number_input ("Chunk Overlap",value= 100)
        temperature_value = st.slider('Temperature', 0.0, 1.0, 0.1)
        k_similarity_num = st.number_input ("K value",value= 2)
        k_similarity = int(k_similarity_num)
        max_output_tokens = st.number_input ("Max Output Tokens",value=25)

    with st.sidebar.expander("📂 Manage Repos", expanded=False):
        for i, repo in enumerate(kr_repos_list):
            new_name = st.text_input(f"Rename {repo}:", value=repo, key=f"new_name_{repo}")
            if new_name and is_valid_repo_name(new_name):
                kr_repos_list[i] = new_name
            elif new_name:
                st.error("Invalid name. Use only alphanumeric and some special characters.")

        if st.button('Update Repository Names'):
            update_repository_list(kr_repos_list)
            
        repo_to_delete = st.radio(f'Delete contents of:', kr_repos_list)
        if st.button('Delete contents'):
            delete_repository_docs(repo_to_delete)
        

    task= None
    task = st.sidebar.radio('Choose task:', task_list, help = "Program can both Load Data and perform query", index=0)
    selected_sources = None
    summarize = None
    macro_view = None
    youtube_url = None
    uploaded_files = None
    trigger_inference = None
    upload_kr_docs_button = None
    repo_selected_for_upload = None
    kr_repos_chosen = None
    selected_sources_image = None
    # Initialize session state variables if they don't exist
    if 'repo_selected_for_upload' not in st.session_state:
        st.session_state['repo_selected_for_upload'] = 'Repo1'
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = None

    if (task == 'Data Load'):
            
            print ('Data Load triggered, assiging ingestion source')
            repo_selected_for_upload = st.radio('Choose Knowledge Repository:', st.session_state['kr_repos_list'], help = "For loading documents into specific domain")
                # Check if the repo selection has changed
            if repo_selected_for_upload != st.session_state['repo_selected_for_upload']:
                # Clear the uploaded files if the repo has changed
                st.session_state['uploaded_files'] = None
                # Update the current repo selection
                st.session_state['repo_selected_for_upload'] = repo_selected_for_upload
            
            uploaded_files = st.file_uploader(f"Upload files for {repo_selected_for_upload}", accept_multiple_files=True)
            privacy_setting = st.radio("Select file privacy level:", ('Private', 'Public', 'Protected'))
            upload_kr_docs_button = st.button("Upload", key="upload_kr_docs")
           
    elif (task == 'Query'):
            print ('In Query')
            selected_sources_image = st.sidebar.multiselect(
                'Image Generation:',
                text2Image_source               
              )
            trigger_inference = st.sidebar.checkbox("Run Image Inference")

            
            if 'text2Image' in selected_sources_image:
                selected_sources = []
            else:    
        
                selected_sources = st.sidebar.multiselect(
                    'Sources:',
                    other_sources,
                    other_sources_default
                )
                if 'KR' in selected_sources:
                
                    kr_repos_chosen = st.sidebar.multiselect(
                        'KR:',
                        st.session_state['kr_repos_list'],
                        st.session_state['kr_repos_list'][0]  # Default selection
                    )
                else:
                    
                    kr_repos_chosen = None
                macro_view = st.sidebar.checkbox("Macro View", value=False, key=None, help='If checked, the full input would be passed to the model, Use GPT4 32k or better', on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
                        
                if len(selected_sources) > 1:
                    summarize = st.sidebar.checkbox("Summarize", value=False, key=None, help='If checked, summarizes content from all sources along with individual responses', on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
                else:                
                    summarize = None

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
        youtube_url,
        k_similarity,
        kr_repos_chosen,
        task,
        upload_kr_docs_button,
        repo_selected_for_upload,
        st.session_state['kr_repos_list'],
        embedding_model_name,
        selected_sources_image,
        macro_view,
        int (max_output_tokens),
        chunk_size,
        chunk_overlap,
        temperature_value,
        show_text_area,
        trigger_inference,
        domain_choice,
        privacy_setting
        
    )
