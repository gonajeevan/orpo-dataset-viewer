import streamlit as st
import pandas as pd
import json
import os

# URL of the Parquet dataset
url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet"

# Function to load the dataset and cache it
@st.cache
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

# Sample the data before filtering
sampled_data = data.sample(n=5000, random_state=1)

# Get unique data source types from the entire dataset
data_source_types = data['source'].unique()

# Streamlit app
st.title("ORPO Dataset Viewer (5k Samples)")

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

# Load viewed questions history and comments
viewed_history_and_comments = load_viewed_history_and_comments()

# Allow the user to select the data source type
selected_data_source = st.selectbox("Select Data Sub-Source Type", data_source_types)

# Filter the sampled dataset based on the selected data source type
filtered_data = sampled_data[sampled_data['source'] == selected_data_source]

# Function to parse the chosen and rejected responses with error handling
def format_conversation(conversation):
    formatted_conversation = ""
    for entry in conversation:
        role = entry.get('role', 'unknown')
        content = entry.get('content', 'No content field found')
        formatted_conversation += f"**<<{role.capitalize()}>>**:\n{content}\n\n"
    return formatted_conversation.strip()

# def format_conversation(conversation):
#     formatted_conversation = ""
#     for entry in conversation:
#         role = entry.get('role', 'unknown')
#         content = entry.get('content', 'No content field found')
#         if role.strip().lower() == 'user':
#             formatted_conversation += f"<span style='color:yellow'><strong><<{role.capitalize()}>>:</strong></span>\n{content}\n\n"
#         elif role.strip().lower() == 'assistant':
#             formatted_conversation += f"<span style='color:green'><strong><<{role.capitalize()}>>:</strong></span>\n{content}\n\n"
#         else:
#             formatted_conversation += f"<strong><<{role.capitalize()}>>:</strong>\n{content}\n\n"
#     return formatted_conversation.strip()

# Apply the parsing function to the chosen and rejected columns
filtered_data['chosen_response'] = filtered_data['chosen'].apply(format_conversation)
filtered_data['rejected_response'] = filtered_data['rejected'].apply(format_conversation)

# Create a list of question indices with their viewed status
question_options = [
    f"{i} - {'Viewed' if i in viewed_history_and_comments.get('viewed', []) else 'Not Viewed'}"
    for i in filtered_data.index
]

# Create a dropdown for selecting the question index
index_selection = st.selectbox("Select a Question Index", question_options)

# Extract the actual index from the selected option
selected_index = int(index_selection.split(" - ")[0])

# Get the selected question's details
selected_data = filtered_data.loc[selected_index]

# Update the session state with the new viewed question index
if 'viewed' not in viewed_history_and_comments:
    viewed_history_and_comments['viewed'] = []

if selected_index not in viewed_history_and_comments['viewed']:
    viewed_history_and_comments['viewed'].append(selected_index)

# Display the details
st.markdown(f"### Data Sub-Source Type: **{selected_data['source']}**")

st.markdown("### Question:")
st.markdown(f"**{selected_data['prompt']}**")

st.markdown("### Chosen Response:")
# st.markdown(selected_data['chosen_response'], unsafe_allow_html=True)
st.markdown(selected_data['chosen_response'])

st.markdown("### Rejected Response:")
# st.markdown(selected_data['rejected_response'], unsafe_allow_html=True)
st.markdown(selected_data['rejected_response'])

# Display comments section
st.markdown("### Comments")
comment_section = viewed_history_and_comments.get('comments', {})

# Display all comments for the selected question
all_comments = []
for user, comments in comment_section.items():
    if str(selected_index) in comments:
        all_comments.append(f"**{user}**:\n{comments[str(selected_index)]}\n\n")
st.markdown("\n".join(all_comments) if all_comments else "No comments yet.")

# Comment input section
if 'anonymous' not in comment_section:
    comment_section['anonymous'] = {}

comment = st.text_area("Enter your comments here:", value=comment_section['anonymous'].get(str(selected_index), ""))
if st.button("Save Comment"):
    comment_section['anonymous'][str(selected_index)] = comment
    viewed_history_and_comments['comments'] = comment_section
    save_viewed_history_and_comments(viewed_history_and_comments)
    st.success("Comment saved!")

# Save the updated viewed history and comments
save_viewed_history_and_comments(viewed_history_and_comments)
