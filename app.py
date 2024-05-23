import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import requests
from io import BytesIO

@st.cache
def load_data():
    url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet"
    response = requests.get(url)
    data = pd.read_parquet(BytesIO(response.content))
    return data

data = load_data()

# Streamlit app
st.title('ORPO-DPO-MIX-40K Dataset Viewer')

# Select a question to view
question_index = st.slider('Select question index', 0, len(data) - 1, 0)
selected_row = data.iloc[question_index]

st.write(f"**Source**: {selected_row['source']}")
st.write(f"**Prompt**: {selected_row['prompt']}")

st.subheader("Chosen Response")
st.json(selected_row['chosen'])

st.subheader("Rejected Response")
st.json(selected_row['rejected'])

# Visualization
st.sidebar.title("Dataset Visualization")
if st.sidebar.checkbox("Show Dataset Summary"):
    st.sidebar.write(data.describe(include='all'))

if st.sidebar.checkbox("Show Source Distribution"):
    st.sidebar.bar_chart(data['source'].value_counts())
