import os
import dotenv
import boto3
from openai import OpenAI
env_vars = dotenv.dotenv_values()
for key in env_vars:
    os.environ[key] = env_vars[key]

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ORGANIZATION = os.getenv('OPENAI_ORGANIZATION')
client = OpenAI(organization=OPENAI_ORGANIZATION, api_key=OPENAI_API_KEY)

def upload_to_s3_refactor(bucket_name, uploaded_file):
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    aws_bucket_input_path = os.getenv('S3_BUCKET_INPUT_PATH')

    # Create an S3 client
    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
    s3_target_path = aws_bucket_input_path + uploaded_file.name
    s3.put_object(Body=uploaded_file.read(), Bucket=bucket_name, Key=s3_target_path)
    return s3_target_path

def generate_presigned_url(bucket_name, s3_path):
    s3_client = boto3.client('s3')
    return s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': s3_path}, ExpiresIn=3600)

def get_inference(question, image_url):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content if response.choices else None