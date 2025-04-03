import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import pandas as pd

# Load the .env file
load_dotenv()

# Get token from environment variable
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

# Login to Hugging Face
login(token=hf_token)

# Load the dataset
dataset = load_dataset("HackerNoon/tech-company-news-data-dump")

# Convert to DataFrame and filter first 100 rows
df = dataset["train"].to_pandas().head(100)

# Filter for rows containing 'Samsung' or 'Google' in 'companyName' or 'title'
mask = (
    df["companyName"].str.contains("Samsung", case=False, na=False) |
    df["companyName"].str.contains("Google", case=False, na=False) |
    df["title"].str.contains("Samsung", case=False, na=False) |
    df["title"].str.contains("Google", case=False, na=False)
)
filtered_df = df.loc[mask, ["companyName", "title", "description"]]

# Save the filtered DataFrame to a CSV file
filtered_df.to_csv("filtered_news_samsung_google.csv", index=False)

print("Filtered CSV saved as 'filtered_news_samsung_google.csv'")

