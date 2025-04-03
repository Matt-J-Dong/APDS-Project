import pandas as pd

# Load the CSV
df = pd.read_csv("filtered_news_top_companies.csv")

# Drop the 'companyName' column
df = df.drop(columns=["companyName"])

# Drop rows where all values are NaN or where title/published_at are missing
df = df.dropna(subset=["title", "published_at"])

# Save cleaned data to new file
df.to_csv("filtered_news_hugging_face.csv", index=False)

print("✅ Cleaned data saved as 'filtered_news_hugging_face.csv'")
