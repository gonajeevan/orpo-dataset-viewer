import pandas as pd

# Load the dataset from the Parquet file
url = "https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k/resolve/main/data/train-00000-of-00001.parquet"
data = pd.read_parquet(url, engine='pyarrow')

# Display a sample of the raw data for chosen and rejected columns
chosen_sample = data['chosen'].head(10)
rejected_sample = data['rejected'].head(10)

print("Sample of the 'chosen' column:")
print(chosen_sample)

print("\nSample of the 'rejected' column:")
print(rejected_sample)
