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
st.title("Response Visualization")

# Author details
st.markdown("""
**App Author:**
- Jeevan
""")

# Dataset credits
st.markdown("""
**Dataset Credits:**
- **Author:** mlabonne
- **Source:** [ORPO-DPO-MIX-40K Dataset on Hugging Face](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/tree/main/data)
""")

# User name input
username = st.text_input("Enter your username")

if username:
    # Load viewed questions history and comments
    viewed_history_and_comments = load_viewed_history_and_comments()

    # Initialize user's viewed questions and comments if not already present
    if username not in viewed_history_and_comments:
        viewed_history_and_comments[username] = {'viewed': [], 'comments': {}}

    # Allow the user to select the data source type
    selected_data_source = st.selectbox("Select Data Source Type", data_source_types)

    # Filter the sampled dataset based on the selected data source type
    filtered_data = sampled_data[sampled_data['source'] == selected_data_source]

    # Function to parse the chosen and rejected responses with error handling
    def format_conversation(conversation):
        formatted_conversation = ""
        for entry in conversation:
            role = entry.get('role', 'unknown')
            content = entry.get('content', 'No content field found')
            formatted_conversation += f"**<{role.capitalize()}>**:\n{content}\n\n"
        return formatted_conversation.strip()

    # Apply the parsing function to the chosen and rejected columns
    filtered_data['chosen_response'] = filtered_data['chosen'].apply(format_conversation)
    filtered_data['rejected_response'] = filtered_data['rejected'].apply(format_conversation)

    # Create a list of question indices with their viewed status
    question_options = [
        f"{i} - {'Viewed' if i in viewed_history_and_comments[username]['viewed'] else 'Not Viewed'}"
        for i in filtered_data.index
    ]

    # Create a dropdown for selecting the question index
    index_selection = st.selectbox("Select a Question Index", question_options)

    # Extract the actual index from the selected option
    selected_index = int(index_selection.split(" - ")[0])

    # Get the selected question's details
    selected_data = filtered_data.loc[selected_index]

    # Update the session state with the new viewed question index
    if selected_index not in viewed_history_and_comments[username]['viewed']:
        viewed_history_and_comments[username]['viewed'].append(selected_index)

    # Display the details
    st.markdown("### Data Source Type:")
    st.markdown(f"**{selected_data['source']}**")

    st.markdown("### Question:")
    st.markdown(f"**{selected_data['prompt']}**")

    st.markdown("### Chosen Response:")
    st.markdown(selected_data['chosen_response'])

    st.markdown("### Rejected Response:")
    st.markdown(selected_data['rejected_response'])

    # Add a checkbox for the author to view all comments
    view_all_comments = st.checkbox("View all comments")

    # Display comments section
    st.markdown("### Your Comments")
    
    if view_all_comments:
        # Display all comments for the selected question
        all_comments = []
        for user, data in viewed_history_and_comments.items():
            if str(selected_index) in data['comments']:
                all_comments.append(f"**{user}**:\n{data['comments'][str(selected_index)]}\n\n")
        st.markdown("\n".join(all_comments) if all_comments else "No comments yet.")
    else:
        # Comment input section
        comment = st.text_area("Enter your comments here:", value=viewed_history_and_comments[username]['comments'].get(str(selected_index), ""))
        if st.button("Save Comment"):
            viewed_history_and_comments[username]['comments'][str(selected_index)] = comment
            save_viewed_history_and_comments(viewed_history_and_comments)
            st.success("Comment saved!")

    # Save the updated viewed history and comments
    save_viewed_history_and_comments(viewed_history_and_comments)
else:
    st.write("Please enter dummy username to continue.(To track history and for no other use)")
