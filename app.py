import streamlit as st
import pandas as pd
import json
import os
import difflib

# URL of the Parquet dataset
url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet"

# Function to load the dataset and cache it
@st.cache_data
def load_data():
    data = pd.read_parquet(url, engine='pyarrow')
    return data

# Load the dataset
data = load_data()

# Get unique data source types from the entire dataset
data_source_types = data['source'].unique()

# Streamlit app
st.title("ORPO Dataset Viewer")

# Dataset credits
st.markdown(f"")
st.markdown("""
**Dataset Credits:**
- **Source:** [ORPO-DPO-MIX-40K Dataset on Hugging Face](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/tree/main/data)
- **Author:** Maxime Labonne (mlabonne)
""")

# Author details
st.markdown(f"")
st.markdown(f"##### App Author: Jeevan")
st.markdown(f"")
st.markdown(f"")

# Display the number of questions
st.markdown(f"### No of Questions: {data.shape[0]}")

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

# Apply the parsing function to the chosen and rejected columns
filtered_data['chosen_response'] = filtered_data['chosen'].apply(format_conversation)
filtered_data['rejected_response'] = filtered_data['rejected'].apply(format_conversation)

# Create a dropdown for selecting the question index
index_selection = st.selectbox("Select a Question Index", filtered_data.index)

# Get the selected question's details
selected_data = filtered_data.loc[index_selection]

# Display the details
# st.markdown("### Data Source Type:")
# st.markdown(f"**{selected_data['source']}**")
if selected_data_source.lower() in ('toxic-dpo-v0.2'):
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


st.markdown(f"")
st.markdown(f"")
st.markdown("### Question:")
st.markdown(f"**{selected_data['prompt']}**")

# Create tabs for Chosen Response and Rejected Response
tab1, tab2 = st.tabs(["Chosen Response", "Rejected Response"])

with tab1:
    chosen_diff, rejected_diff = highlight_differences(selected_data['chosen_response'], selected_data['rejected_response'])
    st.markdown(chosen_diff, unsafe_allow_html=True)

with tab2:
    chosen_diff, rejected_diff = highlight_differences(selected_data['chosen_response'], selected_data['rejected_response'])
    st.markdown(rejected_diff, unsafe_allow_html=True)

st.markdown("""
<div style="border-top: 2px solid white; margin-top: 20px; padding-top: 10px; text-align: center; font-size: 20px;">
</div>
""", unsafe_allow_html=True)
