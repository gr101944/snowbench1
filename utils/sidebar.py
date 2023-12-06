import os
import dotenv
dotenv.load_dotenv(".env")
env_vars = dotenv.dotenv_values()
for key in env_vars:
    os.environ[key] = env_vars[key]
STATIC_ASSEST_BUCKET_URL = os.getenv("STATIC_ASSEST_BUCKET_URL")
STATIC_ASSEST_BUCKET_FOLDER = os.getenv("STATIC_ASSEST_BUCKET_FOLDER")
LOGO_NAME = os.getenv("LOGO_NAME")    

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
    
    source_data_list = [ 'C&SB', 'TE', 'GNT']
    source_data_list_default = [ 'C&SB']
    task_list = [ 'Data Load', 'Query']
    source_category = [ 'C&SB', 'TE', 'GNT']
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
        model_name  = st.selectbox(label='LLM:', options=model_options, help='GPT-4 in waiting list 🤨')
        embedding_model_name  = st.radio('Embedding:', options=embedding_options, help='Option to change embedding model, keep in mind to match with the LLM 🤨')
        persistence_choice = st.radio('Persistence', persistence_options, help = "Using Pinecone...")
        chunk_size = st.number_input ("Chunk Size",value= 400)
        chunk_overlap = st.number_input ("Chunk Overlap",value= 20)
        temperature_value = st.slider('Temperature', 0.0, 1.0, 0.1)
        k_similarity_num = st.number_input ("K value",value= 5)
        k_similarity = int(k_similarity_num)
        max_output_tokens = st.number_input ("Max Output Tokens",value=512)
 
    task= None
    task = st.sidebar.radio('Choose task:', task_list, help = "Program can both Load Data and perform query", index=0)
    selected_sources = None
    summarize = None
    macro_view = None
    youtube_url = None
    uploaded_files = None
    upload_kr_docs_button = None
    ingest_source_chosen = None
    sources_chosen = None
    selected_sources_image = None

    if (task == 'Data Load'):
            print ('Data Load triggered, assiging ingestion source')
            ingest_source_chosen = st.radio('Choose Knowledge Repository:', source_data_list, help = "For loading documents into specific domain")
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
        sources_chosen,
        task,
        upload_kr_docs_button,
        ingest_source_chosen,
        source_data_list,
        source_category,
        embedding_model_name,
        selected_sources_image,
        macro_view,
        int (max_output_tokens),
        chunk_size,
        chunk_overlap,
        temperature_value
    )
