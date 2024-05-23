import streamlit as st
import pandas as pd

# URL of the Parquet dataset
url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet"

# Function to load the dataset and cache it
@st.cache
def load_data():
    data = pd.read_parquet(url, engine='pyarrow')
    return data

# Load the dataset
data = load_data()

# Get unique data source types
data_source_types = data['source'].unique()

# Streamlit app
st.title("Response Visualization")

# Dataset credits
st.markdown("""
**Dataset Credits:**
- **Source:** [ORPO-DPO-MIX-40K Dataset on Hugging Face](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/tree/main/data)
- **Author:** mlabonne
""")

# Allow the user to select the data source type
selected_data_source = st.selectbox("Select Data Source Type", data_source_types)

# Filter the dataset based on the selected data source type
filtered_data = data[data['source'] == selected_data_source].sample(n=5000, random_state=1)

# Function to parse the chosen and rejected responses with error handling
def format_conversation(conversation):
    formatted_conversation = ""
    for entry in conversation:
        role = entry.get('role', 'unknown')
        content = entry.get('content', 'No content field found')
        formatted_conversation += f"**{role.capitalize()}**:\n{content}\n\n"
    return formatted_conversation.strip()

# Apply the parsing function to the chosen and rejected columns
filtered_data['chosen_response'] = filtered_data['chosen'].apply(format_conversation)
filtered_data['rejected_response'] = filtered_data['rejected'].apply(format_conversation)

# Initialize session state for viewed questions
if 'viewed_questions' not in st.session_state:
    st.session_state.viewed_questions = []

# Create a list of question indices with their viewed status
question_options = [
    f"{i} - {'Viewed' if i in st.session_state.viewed_questions else 'Not Viewed'}"
    for i in filtered_data.index
]

# Create a dropdown for selecting the question index
index_selection = st.selectbox("Select a Question Index", question_options)

# Extract the actual index from the selected option
selected_index = int(index_selection.split(" - ")[0])

# Get the selected question's details
selected_data = filtered_data.loc[selected_index]

# Update the session state with the new viewed question index
if selected_index not in st.session_state.viewed_questions:
    st.session_state.viewed_questions.append(selected_index)

# Display the details
st.markdown("### Data Source Type:")
st.markdown(f"**{selected_data['source']}**")

st.markdown("### Question:")
st.markdown(f"**{selected_data['prompt']}**")

st.markdown("### Chosen Response:")
st.markdown(selected_data['chosen_response'])

st.markdown("### Rejected Response:")
st.markdown(selected_data['rejected_response'])
