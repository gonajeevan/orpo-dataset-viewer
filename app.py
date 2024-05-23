import streamlit as st
import pandas as pd

# URL of the Parquet dataset
url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet"

# Function to load the dataset and cache it
@st.cache
def load_data():
    data = pd.read_parquet(url, engine='pyarrow').sample(n=5000, random_state=1)
    return data

# Load the dataset
data = load_data()

# Function to parse the chosen and rejected responses with error handling
def format_conversation(conversation):
    formatted_conversation = ""
    for entry in conversation:
        role = entry.get('role', 'unknown')
        content = entry.get('content', 'No content field found')
        formatted_conversation += f"**{role.capitalize()}**:\n{content}\n\n"
    return formatted_conversation.strip()

# Apply the parsing function to the chosen and rejected columns
data['chosen_response'] = data['chosen'].apply(format_conversation)
data['rejected_response'] = data['rejected'].apply(format_conversation)

# Initialize session state for viewed questions
if 'viewed_questions' not in st.session_state:
    st.session_state.viewed_questions = []

# Streamlit app
st.title("Response Visualization")

# Dataset credits
st.markdown("""
**Dataset Credits:**
- **Source:** [ORPO-DPO-MIX-40K Dataset on Hugging Face](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/tree/main/data)
- **Author:** mlabonne
""")

# Create a list of question indices with their viewed status
question_options = [
    f"{i} - {'Viewed' if i in st.session_state.viewed_questions else 'Not Viewed'}"
    for i in data.index
]

# Create a dropdown for selecting the question index
index_selection = st.selectbox("Select a Question Index", question_options)

# Extract the actual index from the selected option
selected_index = int(index_selection.split(" - ")[0])

# Get the selected question's details
selected_data = data.loc[selected_index]

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
