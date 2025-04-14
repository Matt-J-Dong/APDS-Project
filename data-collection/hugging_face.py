import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import pandas as pd

# Load the .env file
print("🔄 Loading .env file...")
load_dotenv()

# Get token from environment variable
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("❌ HF_TOKEN not found in .env file")

# Login to Hugging Face
print("🔐 Logging in to Hugging Face...")
login(token=hf_token)

# Load the dataset
print("📦 Loading HackerNoon dataset...")
dataset = load_dataset("HackerNoon/tech-company-news-data-dump")

# Convert to DataFrame and filter first 10,000 rows
print("📄 Converting to DataFrame...")
df = dataset["train"].to_pandas()

# Convert published_at to UTC Zulu time
print("🕒 Converting published_at to UTC Zulu format...")
df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
df["published_at"] = df["published_at"].dt.tz_localize("UTC")
df["published_at"] = df["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# Companies to search for
companies = [
    "Apple",
    "Alphabet",
    "Google",
    "Microsoft",
    "Meta",
    "Nvidia",
    "Tesla",
    "Samsung",
    "Amazon",
    "Oracle",
    "Salesforce",
    "IBM",
    "Netflix",
    "Sony",
    "Cisco",
    "Adobe",
    "Lenovo",
    "Dell",
    "HP",
    "Intel",
    "AMD",
]

# Build the inclusion mask dynamically
print("🔍 Building company search filter...")
inclusion_mask = pd.Series(False, index=df.index)
for company in companies:
    inclusion_mask |= df["companyName"].str.contains(company, case=False, na=False)
    inclusion_mask |= df["title"].str.contains(company, case=False, na=False)

# Exclude "googleapis"
exclusion_mask = ~df["companyName"].str.contains("googleapis", case=False, na=False)

# Final filter
mask = inclusion_mask & exclusion_mask

# Apply filter and select desired columns
filtered_df = df.loc[mask, ["companyName", "title", "published_at"]]

# Save to CSV
print(f"💾 Saving {len(filtered_df)} filtered rows to CSV...")
filtered_df.to_csv("filtered_news_top_companies.csv", index=False)

print("✅ Done! File saved as 'filtered_news_top_companies.csv'")
