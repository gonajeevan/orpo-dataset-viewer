import streamlit as st
import pandas as pd
import difflib
from datasets import load_dataset

# URL of the Parquet dataset
url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet"

# Function to load and cache the dataset
@st.cache_data
def load_data():
    # Load the dataset in streaming mode
    dataset = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train", streaming=True)
    # Convert the streaming dataset to a pandas DataFrame
    data = pd.DataFrame(dataset)
    data_source_types = data['source'].unique()
    return data, data_source_types

# Load the dataset and get unique data source types
data, data_source_types = load_data()

# Streamlit app
st.title("ORPO Dataset Viewer")

# Dataset credits
st.markdown("""
**Dataset Credits:**
- **Source:** [ORPO-DPO-MIX-40K Dataset on Hugging Face](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/tree/main/data)
- **Author:** Maxime Labonne (mlabonne)
""")

# Author details
st.markdown("#### App Author: Jeevan")

# Display the number of questions
st.markdown(f"### Number of Questions: {data.shape[0]}")

# Allow the user to select the data source type
selected_data_source = st.selectbox("Select Data Source Type", data_source_types)

# Filter the dataset based on the selected data source type
filtered_data = data[data['source'] == selected_data_source]

# Function to parse the chosen and rejected responses with error handling
def format_conversation(conversation):
    formatted_conversation = ""
    for entry in conversation:
        role = entry.get('role', 'unknown')
        content = entry.get('content', 'No content field found')
        formatted_conversation += f"<< {role.capitalize()} >>:\n{content}\n\n"
    return formatted_conversation.strip()

# Function to highlight differences between two texts
def highlight_differences(chosen, rejected):
    differ = difflib.ndiff(chosen.splitlines(), rejected.splitlines())
    highlighted_chosen = []
    highlighted_rejected = []

    for line in differ:
        if line.startswith('  '):
            highlighted_chosen.append(line[2:])
            highlighted_rejected.append(line[2:])
        elif line.startswith('- '):
            highlighted_chosen.append(f"<span style='background-color: #ccffcc; color: black;'>{line[2:]}</span>")
        elif line.startswith('+ '):
            highlighted_rejected.append(f"<span style='background-color: #ffcccc; color: black;'>{line[2:]}</span>")

    highlighted_chosen = '\n'.join(highlighted_chosen)
    highlighted_rejected = '\n'.join(highlighted_rejected)

    return highlighted_chosen, highlighted_rejected

# Create a dropdown for selecting the question index
index_selection = st.selectbox("Select a Question Index", filtered_data.index)

# Get the selected question's details
selected_data = filtered_data.loc[index_selection]

# Display the details
if selected_data_source.lower() == 'toxic-dpo-v0.2':
    st.markdown("""
    <div style='border: 1px solid white; padding: 10px;'>
        <em>(From Data Owner)</em>
        <br><br>
        Note that ORPO-DPO-mix-40k contains a dataset (toxic-dpo-v0.2) designed to prompt the model to answer illegal questions. You can remove it as follows:
        <br><br>
        <code>dataset = load_dataset('mlabonne/orpo-mix-40k', split='train')<br></code>
        <code>dataset = dataset.filter(lambda r: r["source"] != "toxic-dpo-v0.2")</code>
        <br><br>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Question:")
st.markdown(f"**{selected_data['prompt']}**")

# Create tabs for Chosen Response and Rejected Response
tab1, tab2 = st.tabs(["Chosen Response", "Rejected Response"])

chosen_response = format_conversation(selected_data['chosen'])
rejected_response = format_conversation(selected_data['rejected'])
chosen_diff, rejected_diff = highlight_differences(chosen_response, rejected_response)

with tab1:
    st.markdown(chosen_diff, unsafe_allow_html=True)

with tab2:
    st.markdown(rejected_diff, unsafe_allow_html=True)

st.markdown("""
<div style="border-top: 2px solid white; margin-top: 20px; padding-top: 10px; text-align: center; font-size: 20px;">
</div>
""", unsafe_allow_html=True)
