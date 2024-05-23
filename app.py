import streamlit as st
import pandas as pd
import json

# URL of the Parquet dataset
url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet"

# Function to load the dataset and cache it
@st.cache_data
def load_data():
    data = pd.read_parquet(url, engine='pyarrow').sample(n=5000, random_state=1)
    return data

# Load the dataset
data = load_data()

# Function to parse the chosen and rejected responses with error handling
def parse_response(response):
    try:
        parsed = json.loads(response.replace("'", "\""))
        if isinstance(parsed, list) and len(parsed) > 0:
            return "\n\n".join([entry.get('content', 'No content field found') for entry in parsed])
        else:
            return "Unexpected JSON format"
    except (json.JSONDecodeError, IndexError, TypeError) as e:
        return f"Invalid response format: {e}"

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

# Select a prompt to display its details
prompt_selection = st.selectbox("Select a Prompt", data['prompt'].unique())

# Filter the dataframe based on the selected prompt
selected_data = data[data['prompt'] == prompt_selection]

# Display the details
if not selected_data.empty:
    st.markdown("### Data Source Type:")
    st.write(selected_data['source'].values[0])

    st.markdown("### Question:")
    st.write(selected_data['prompt'].values[0])

    st.markdown("### Chosen Response:")
    st.write(selected_data['chosen_response'].values[0])

    st.markdown("### Rejected Response:")
    st.write(selected_data['rejected_response'].values[0])
else:
    st.write("No responses available for the selected prompt.")
