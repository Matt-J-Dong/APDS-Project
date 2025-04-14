import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

# Replace with your actual Twitter Bearer Token or set it as an environment variable.
bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")

# Initialize the Tweepy client for Twitter API v2.
client = tweepy.Client(bearer_token=bearer_token)

# Define the search query to fetch tweets that mention "Samsung".
query = "Samsung -is:retweet"  # Exclude retweets for clarity (optional)

# Request tweet fields: created_at for the timestamp and public_metrics for like counts.
tweet_fields = ["created_at", "public_metrics"]

# Search for recent tweets (adjust max_results as needed; max is 100 per request).
response = client.search_recent_tweets(query=query, tweet_fields=tweet_fields, max_results=10)

if response.data:
    for tweet in response.data:
        # 'created_at' is already in UTC, format it as a Zulu time string.
        tweet_time = tweet.created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        like_count = tweet.public_metrics.get("like_count", 0)
        
        print("Tweet:")
        print(tweet.text)
        print("Posted at (UTC Zulu):", tweet_time)
        print("Likes:", like_count)
        print("-" * 40)
else:
    print("No tweets found for the given query.")
