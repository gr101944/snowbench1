import boto3
import os
import openai
import streamlit as st
import dotenv
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
        
        file_name = "image" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".png"
        dotenv.load_dotenv(".env")
        env_vars = dotenv.dotenv_values()
        for key in env_vars:
            os.environ[key] = env_vars[key]
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_DEFAULT_REGION')
        
            # Create an S3 client
        
        aws_bucket = os.getenv('S3_PUBLIC_ACCESS')
        aws_bucket_input_path = os.getenv('S3_PUBLIC_ACCESS_PATH')
        s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
        image_data = requests.get(image_url).content
        s3.put_object(Body=image_data, Bucket=aws_bucket, Key=aws_bucket_input_path + "/" + file_name)

        st.image(image_url, caption="Generated Image", use_column_width=True)
    else:
        st.write("No data available to generate an image.")