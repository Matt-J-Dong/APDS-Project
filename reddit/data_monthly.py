import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# Define function to get the Unix timestamp for the first and last second of a given month
def get_month_timestamps(year, month):
    start_date = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    if month == 12:
        end_date = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc) - timedelta(seconds=1)
    else:
        end_date = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc) - timedelta(seconds=1)
    
    return int(start_date.timestamp()), int(end_date.timestamp())

# Initialize an empty list to store the formatted data
formatted_data = []

# Define the subreddit to fetch data from
subreddit = "apple"

# Set the start date (January 2024) and today's date
start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
today = datetime.now(timezone.utc)

# Loop through each month from January 2024 until today
while start_date < today:
    year, month = start_date.year, start_date.month
    since, until = get_month_timestamps(year, month)

    # Define the API URL with dynamic timestamps
    url = f"https://api.pullpush.io/reddit/submission/search?html_decode=True&subreddit={subreddit}&since={since}&until={until}&size=100"

    # Make the GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Convert response to JSON
        submissions = data.get("data", [])  # Extract the list of submissions

        # Process the data: Extract relevant fields and format timestamps
        for submission in submissions:
            created_utc = submission.get("created")  # Unix timestamp
            title = submission.get("title", "No Title")  # Post title

            # Convert Unix timestamp to UTC Zulu format
            if created_utc is not None:
                time_data = datetime.fromtimestamp(created_utc, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                time_data = "Unknown"

            formatted_data.append({"Time Data": time_data, "Headline": title})

        print(f"Fetched {len(submissions)} submissions for {year}-{month:02d}")

    else:
        print(
            f"Error fetching data for {year}-{month:02d}: {response.status_code}, {response.text}"
        )

    # Move to the next month
    if month == 12:
        start_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        start_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Save to CSV
csv_filename = "reddit_posts_monthly.csv"
df.to_csv(csv_filename, index=False)

print(f"All data saved to {csv_filename}")
