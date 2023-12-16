import os
import dotenv
dotenv.load_dotenv(".env")
env_vars = dotenv.dotenv_values()
for key in env_vars:
    os.environ[key] = env_vars[key]
STATIC_ASSEST_BUCKET_URL = os.getenv("STATIC_ASSEST_BUCKET_URL")
STATIC_ASSEST_BUCKET_FOLDER = os.getenv("STATIC_ASSEST_BUCKET_FOLDER")
LOGO_NAME = os.getenv("LOGO_NAME")  
import pinecone

import streamlit as st
import pinecone

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


    # Set top_k to a large number, assuming the maximum number of documents in a repo
 


  

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
    html_code = """
        <style>
            div.row-widget.stRadio > div{flex-direction:row;}
        </style>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    
    image_path = os.path.join(STATIC_ASSEST_BUCKET_URL, STATIC_ASSEST_BUCKET_FOLDER, LOGO_NAME)
    with st.sidebar:
        st.image(image_path, width=55)
    
    kr_repos_list = [ 'Repo1', 'Repo2', 'Repo3']
    kr_repos_list_default = [ 'Repo1']
    task_list = [ 'Data Load', 'Query']
 
    other_sources = ['Open AI', 'YouTube', 'Google', 'KR', 'text2Image']
    text2Image_source = ['text2Image']
    other_sources_default = ['KR']
    embedding_options = ['text-embedding-ada-002']
    persistence_options = [ 'Pinecone']
    model_options = ['gpt-3.5-turbo','gpt-4', 'gpt-4-1106-preview','gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k']
    # 
    url = "https://www.persistent.com/"
    
   
    with st.sidebar.expander(" 🛠️ Configurations ", expanded=False):
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
        repo_to_delete = st.radio("Select a Repository to Delete", kr_repos_list)
        if st.button('Delete Selected Repository'):
            delete_repository_docs(repo_to_delete)
 
    task= None
    task = st.sidebar.radio('Choose task:', task_list, help = "Program can both Load Data and perform query", index=0)
    selected_sources = None
    summarize = None
    macro_view = None
    youtube_url = None
    uploaded_files = None
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
            repo_selected_for_upload = st.radio('Choose Knowledge Repository:', kr_repos_list, help = "For loading documents into specific domain")
                # Check if the repo selection has changed
            if repo_selected_for_upload != st.session_state['repo_selected_for_upload']:
                # Clear the uploaded files if the repo has changed
                st.session_state['uploaded_files'] = None
                # Update the current repo selection
                st.session_state['repo_selected_for_upload'] = repo_selected_for_upload
            uploaded_files = st.file_uploader(f"Upload files for {repo_selected_for_upload}", accept_multiple_files=True)
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
           
                selected_sources = st.sidebar.multiselect(
                    'Sources:',
                    other_sources,
                    other_sources_default
                )
                if 'KR' in selected_sources:
                
                    kr_repos_chosen = st.sidebar.multiselect(
                        'KR:',
                        kr_repos_list,
                        kr_repos_list_default
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
        kr_repos_list,
        embedding_model_name,
        selected_sources_image,
        macro_view,
        int (max_output_tokens),
        chunk_size,
        chunk_overlap,
        temperature_value,
        show_text_area
    )
