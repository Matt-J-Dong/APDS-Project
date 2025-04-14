import requests
import pandas as pd
from datetime import datetime, timezone, timedelta


# Define function to get the Unix timestamp for the first and last second of a given week
def get_week_timestamps(start_date):
    end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return int(start_date.timestamp()), int(end_date.timestamp())


# Initialize an empty list to store the formatted data
formatted_data = []

# Define the subreddit to fetch data from
subreddit = "apple"

# Set the start date (January 1, 2024) and today's date
start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
today = datetime.now(timezone.utc)

# Loop through each week from January 1, 2024, until today
while start_date < today:
    since, until = get_week_timestamps(start_date)

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

        print(
            f"Fetched {len(submissions)} submissions for the week starting {start_date.strftime('%Y-%m-%d')}"
        )

    else:
        print(
            f"Error fetching data for the week starting {start_date.strftime('%Y-%m-%d')}: {response.status_code}, {response.text}"
        )

    # Move to the next week
    start_date += timedelta(weeks=1)

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Save to CSV
csv_filename = "reddit_posts_weekly.csv"
df.to_csv(csv_filename, index=False)

print(f"All data saved to {csv_filename}")
