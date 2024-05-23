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

# Function to load viewed questions history and comments from a JSON file
def load_viewed_history_and_comments():
    if os.path.exists("viewed_questions_and_comments.json"):
        with open("viewed_questions_and_comments.json", "r") as file:
            return json.load(file)
    return {}

# Function to save viewed questions history and comments to a JSON file
def save_viewed_history_and_comments(viewed_history_and_comments):
    with open("viewed_questions_and_comments.json", "w") as file:
        json.dump(viewed_history_and_comments, file)

# Load the dataset
data = load_data()

# Get unique data source types from the entire dataset
data_source_types = data['source'].unique()

# Streamlit app
st.title("ORPO Dataset Viewer")

# Dataset credits
st.markdown("""
**Dataset Credits:**
- **Source:** [ORPO-DPO-MIX-40K Dataset on Hugging Face](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/tree/main/data)
- **Author:** mlabonne
""")

# Author details
st.markdown("""
**App Author:**
- Jeevan
""")

# Display the number of questions
st.markdown(f"### No of Questions: {data.shape[0]}")

# Load viewed questions history and comments
viewed_history_and_comments = load_viewed_history_and_comments()

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
            highlighted_chosen.append(f"<span style='background-color: #e6ffe6; color: black;'>{line[2:]}</span>")
        elif line.startswith('+ '):
            highlighted_rejected.append(f"<span style='background-color: #ffe6e6; color: black;'>{line[2:]}</span>")

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
st.markdown("### Data Source Type:")
st.markdown(f"**{selected_data['source']}**")

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

# Display comments section
st.markdown("### Comments")
comment_section = viewed_history_and_comments.get('comments', {})

# Display all comments for the selected question
all_comments = []
for user, comments in comment_section.items():
    if str(index_selection) in comments:
        all_comments.append(f"**{user}**:<br>{comments[str(index_selection)]}<br><br>")
st.markdown("\n".join(all_comments) if all_comments else "No comments yet.", unsafe_allow_html=True)

# Comment input section
if 'anonymous' not in comment_section:
    comment_section['anonymous'] = {}

comment = st.text_area("Enter your comments here:", value=comment_section['anonymous'].get(str(index_selection), ""))
if st.button("Save Comment"):
    comment_section['anonymous'][str(index_selection)] = comment
    viewed_history_and_comments['comments'] = comment_section
    save_viewed_history_and_comments(viewed_history_and_comments)
    st.success("Comment saved!")

# Save the updated viewed history and comments
save_viewed_history_and_comments(viewed_history_and_comments)
