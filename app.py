import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import requests
from io import BytesIO

# Function to load the dataset
@st.cache
def load_data():
    url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet"
    response = requests.get(url)
    data = pd.read_parquet(BytesIO(response.content))
    return data

# Function to get a random sample of data
@st.cache
def get_sample(data, n=5000):
    return data.sample(n)

# Load the full dataset
data = load_data()

# Get a random sample of 5k rows
sampled_data = get_sample(data)

# Streamlit app
st.title('ORPO-DPO-MIX-40K Dataset Viewer')

# Add credits to dataset owner
st.markdown("""
**Dataset Credits:**
- Dataset provided by [mlabonne](https://huggingface.co/mlabonne)
- Dataset URL: [ORPO-DPO-MIX-40K](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)
""")

# Select a question to view
question_index = st.slider('Select question index', 0, len(sampled_data) - 1, 0)
selected_row = sampled_data.iloc[question_index]

st.write(f"**Source**: {selected_row['source']}")
st.write(f"**Prompt**: {selected_row['prompt']}")

st.subheader("Chosen Response")
st.write(selected_row['chosen'])

st.subheader("Rejected Response")
st.write(selected_row['rejected'])

# Visualization
st.sidebar.title("Dataset Visualization")
if st.sidebar.checkbox("Show Dataset Summary"):
    st.sidebar.write(sampled_data.describe(include='all'))

if st.sidebar.checkbox("Show Source Distribution"):
    st.sidebar.bar_chart(sampled_data['source'].value_counts())
