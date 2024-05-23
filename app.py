import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import requests
from io import BytesIO
import json

# URL of the Parquet dataset
url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/orpo_40k_dataset.parquet"

# Function to load the dataset and cache it
@st.cache_data
def load_data():
    data = pd.read_parquet(url, engine='pyarrow').sample(n=5000, random_state=1)
    return data

# Load the dataset
data = load_data()

# Function to parse the chosen and rejected responses
def parse_response(response):
    return json.loads(response)[0]['content'] if response else ""

# Apply the parsing function to the chosen and rejected columns
data['chosen_response'] = data['chosen'].apply(parse_response)
data['rejected_response'] = data['rejected'].apply(parse_response)

# Streamlit app
st.title("Response Visualization")

# Dataset credits
st.markdown("""
**Dataset Credits:**
- **Source:** [ORPO-DPO-MIX-40K Dataset on Hugging Face](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/tree/main/data)
- **Author:** mlabonne
""")

# Select a prompt to display its chosen and rejected responses
prompt_selection = st.selectbox("Select a Prompt", data['prompt'].unique())

# Filter the dataframe based on the selected prompt
selected_data = data[data['prompt'] == prompt_selection]

# Display the chosen and rejected responses
if not selected_data.empty:
    st.write("Chosen Response:")
    st.write(selected_data['chosen_response'].values[0])

    st.write("Rejected Response:")
    st.write(selected_data['rejected_response'].values[0])
else:
    st.write("No responses available for the selected prompt.")
