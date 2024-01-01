from openai import OpenAI
import re
import os
import dotenv
import snowflake.connector
import streamlit as st
import pandas as pd
from prompts3 import get_system_prompt

# Load environment variables
dotenv.load_dotenv(".env")
env_vars = dotenv.dotenv_values()
for key in env_vars:
    os.environ[key] = env_vars[key]

# Fetching OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Function to establish a Snowflake connection
def create_snowflake_connection():
    return snowflake.connector.connect(
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        account=os.environ.get("SNOWFLAKE_ACCOUNT"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        role=os.environ.get("SNOWFLAKE_ROLE")
    )

# Streamlit UI setup
st.title("❄️ + 🛠️ SnowBench")

# Initialize the chat messages history
client = OpenAI(api_key=OPENAI_API_KEY)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": get_system_prompt()}]

# Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the existing chat messages
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "results" in message:
            st.dataframe(message["results"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = ""
        resp_container = st.empty()
        for delta in client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        ):
            response += (delta.choices[0].delta.content or "")
            resp_container.markdown(response)

        message = {"role": "assistant", "content": response}

        # Parse the response for a SQL query and execute if available
        sql_match = re.search(r"```sql\n(.*)\n```", response, re.DOTALL)
        if sql_match:
            sql = sql_match.group(1)

            # Use the custom function to create a Snowflake connection
            with create_snowflake_connection() as conn, st.spinner('Executing query...'):
                cur = conn.cursor()
                cur.execute(sql)
                results = cur.fetchall()

                # Convert results to DataFrame
                if results:
                    df = pd.DataFrame(results, columns=[col[0] for col in cur.description])
                    df.index = df.index + 1  # Adjusting index to start from 1
                    message["results"] = df
                    st.dataframe(df)
                else:
                    st.write("No results found.")

        st.session_state.messages.append(message)
