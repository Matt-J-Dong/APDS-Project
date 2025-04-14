import pandas as pd
from datetime import datetime

# Load the CSV file
df = pd.read_csv("raw_partner_headlines.csv")

# Keep only the specified columns
df = df[['id', 'headline', 'date', 'stock']]

# Convert 'date' to UTC Zulu time format
df['date'] = pd.to_datetime(df['date'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Save full cleaned file
df.to_csv("partner_headlines_cleaned.csv", index=False)

# Split the DataFrame in half
midpoint = len(df) // 2
df1 = df.iloc[:midpoint]
df2 = df.iloc[midpoint:]

# Save the split files
df1.to_csv("partner_headlines_cleaned_1.csv", index=False)
df2.to_csv("partner_headlines_cleaned_2.csv", index=False)
