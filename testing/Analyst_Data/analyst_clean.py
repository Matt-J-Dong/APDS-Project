import pandas as pd

# Load the CSV file
df = pd.read_csv("analyst_ratings_processed.csv")

# Convert 'date' to UTC Zulu time, coerce invalid rows
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Optional: Drop rows where conversion failed
df = df.dropna(subset=['date'])

# Calculate the midpoint to split
midpoint = len(df) // 2

# Split the DataFrame into two
df1 = df.iloc[:midpoint]
df2 = df.iloc[midpoint:]

# Save both parts to new CSV files
df1.to_csv("analyst_ratings_processed_cleaned_1.csv", index=False)
df2.to_csv("analyst_ratings_processed_cleaned_2.csv", index=False)
