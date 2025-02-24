import requests
import pandas as pd
from dotenv import load_dotenv
import os
from tabulate import tabulate

load_dotenv()

client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")
username = os.environ.get("REDDIT_USERNAME")  # Huh wow
password = os.environ.get("PASSWORD")
user_agent = 'Test/0.0.1'

# Set up HTTP Basic Auth with your client ID and secret
auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
data = {"grant_type": "password", "username": username, "password": password}
headers = {"User-Agent": user_agent}
# Request the access token
response = requests.post(
    "https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers
)
token = response.json()["access_token"]
# Update headers with the access token
headers.update({"Authorization": f"bearer {token}"})

"""
Get the first 100 posts from hot posts on Apple subreddit!
"""
res = requests.get(
    "https://oauth.reddit.com/r/apple/hot", headers=headers, params={"limit": 100}
)

rows = []
for post in res.json()["data"]["children"]:
    data_post = post["data"]
    rows.append(
        {
            "id": data_post["id"],
            "subreddit": data_post["subreddit"],
            "title": data_post["title"],
            "selftext": data_post["selftext"],
            "upvote_ratio": data_post["upvote_ratio"],
            "ups": data_post["ups"],
            "downs": data_post["downs"],
            "score": data_post["score"],
            "edited": data_post.get("edited"),
            "all_awardings": data_post.get("all_awardings"),
            "controversiality": data_post.get("controversiality"),
            "num_comments": data_post.get("num_comments"),
            "link_flair_text": data_post.get("link_flair_text"),
            "created_utc": data_post.get("created_utc"),  # Epoch seconds
        }
    )

df = pd.DataFrame(rows)
# Convert epoch time to human-readable UTC Zulu time for posts.
df["created_time"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)
print("Posts DataFrame:")
print(df)
# print(tabulate(df, headers='keys', tablefmt='psql')) # This doesn't work well right now because of weird post formatting.
print("\n" + "="*50 + "\n")

def extract_comments(comments_list, post_id, depth=0):
    extracted = []
    for comment in comments_list:
        # Skip non-comment objects (e.g., "more" objects)
        if comment["kind"] != "t1":
            continue
        data = comment["data"]
        extracted.append(
            {
                "post_id": post_id,
                "comment_id": data.get("id"),
                "parent_id": data.get("parent_id"),
                "author": data.get("author"),
                "body": data.get("body"),
                "ups": data.get("ups"),
                "downs": data.get("downs"),
                "score": data.get("score"),
                "edited": data.get("edited"),
                "all_awardings": data.get("all_awardings"),
                "controversiality": data.get("controversiality"),
                "created_utc": data.get("created_utc"),
                "depth": depth,
            }
        )
        # Process nested replies if they exist
        if data.get("replies") and isinstance(data["replies"], dict):
            child_comments = extract_comments(
                data["replies"]["data"]["children"], post_id, depth=depth + 1
            )
            extracted.extend(child_comments)
    return extracted

all_comments = []

# Only process the first 10 posts to save compute.
for idx, row in df.head(10).iterrows():
    post_id = row["id"]
    comments_url = f"https://oauth.reddit.com/comments/{post_id}"
    response = requests.get(comments_url, headers=headers, params={"limit": 100})
    comments_json = response.json()
    # The second element in the response contains the comments listing
    top_comments = comments_json[1]["data"]["children"]
    extracted_comments = extract_comments(top_comments, post_id)
    all_comments.extend(extracted_comments)

# Create a DataFrame from the extracted comment data
comments_df = pd.DataFrame(all_comments)
# Convert epoch time to human-readable UTC Zulu time for comments.
comments_df["created_time"] = pd.to_datetime(
    comments_df["created_utc"], unit="s", utc=True
).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
print("Combined Comments DataFrame (from first 10 posts):")
print(comments_df.head())
print("\n" + "="*50 + "\n")

# Create separate DataFrames for each post (if needed)
dfs_by_post = {}
for post_id in comments_df["post_id"].unique():
    dfs_by_post[post_id] = comments_df[comments_df["post_id"] == post_id]

for post_id, post_comments_df in dfs_by_post.items():
    print(f"Comments for post {post_id}:")
    print(post_comments_df)
    print("\n" + "-"*40 + "\n")

# --------------------------------------------------
# Save the posts and comments DataFrames to CSV files
# --------------------------------------------------

# Save posts data to 'reddit_posts.csv'
df.to_csv("reddit_posts.csv", index=False)
print("Saved posts to reddit_posts.csv")

# Save comments data to 'reddit_comments.csv'
comments_df.to_csv("reddit_comments.csv", index=False)
print("Saved comments to reddit_comments.csv")

# --------------------------------------------------
# Create a new CSV file with two columns:
#   - created_time: the UTC Zulu time (YYYY-MM-DDTHH:MM:SSZ)
#   - text: contains either the post title, post selftext, or comment body.
# For posts that have both a title and selftext, each becomes a separate row.
# --------------------------------------------------

text_rows = []

# Process posts: include title and selftext as separate data points.
for idx, row in df.iterrows():
    # If title is non-empty, add it.
    if row["title"] and str(row["title"]).strip():
        text_rows.append({"created_time": row["created_time"], "text": row["title"]})
    # If selftext is non-empty, add it.
    if row["selftext"] and str(row["selftext"]).strip():
        text_rows.append({"created_time": row["created_time"], "text": row["selftext"]})

# Process comments: use the comment body.
for idx, row in comments_df.iterrows():
    if row["body"] and str(row["body"]).strip():
        text_rows.append({"created_time": row["created_time"], "text": row["body"]})

# Create the new DataFrame and save to CSV.
text_df = pd.DataFrame(text_rows)
text_df.to_csv("reddit_texts.csv", index=False)
print("Saved combined texts to reddit_texts.csv")
