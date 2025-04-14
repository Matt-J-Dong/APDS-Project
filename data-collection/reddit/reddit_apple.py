import requests
import pandas as pd
from dotenv import load_dotenv
import os
from tqdm import tqdm  # For progress bars
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

load_dotenv()

client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")
username = os.environ.get("REDDIT_USERNAME")  # Huh wow
password = os.environ.get("PASSWORD")
user_agent = 'Test/0.0.1'

# Set up HTTP Basic Auth with your client ID and secret.
auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
data = {"grant_type": "password", "username": username, "password": password}
headers = {"User-Agent": user_agent}

# Request the access token.
response = requests.post(
    "https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers
)
token = response.json()["access_token"]
headers.update({"Authorization": f"bearer {token}"})

subreddit = "apple"
all_posts = []

# Get posts for each month over the past 12 months.
now = datetime.now(timezone.utc)
for i in range(12):
    # Compute the start and end of each month.
    month_end = datetime(now.year, now.month, 1) - relativedelta(months=i)
    month_start = month_end - relativedelta(months=1)
    start_ts = int(month_start.timestamp())
    end_ts = int(month_end.timestamp())

    # Build the CloudSearch query string.
    query = f"timestamp:{start_ts}..{end_ts}"
    search_url = f"https://oauth.reddit.com/r/{subreddit}/hot"
    params = {
        "q": query,
        "sort": "new",
        "restrict_sr": True,
        "syntax": "cloudsearch",
        "limit": 100,  # maximum per call
    }

    print(
        f"Fetching posts from {month_start.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}"
    )
    r = requests.get(search_url, headers=headers, params=params)
    data_json = r.json()
    for post in data_json.get("data", {}).get("children", []):
        p_data = post["data"]
        all_posts.append(
            {
                "id": p_data["id"],
                "subreddit": p_data["subreddit"],
                # Fix text encoding in title and selftext:
                # It turns out that there is no error actually and Excel is just bad at displaying certain characters. I'm keeping this code in anyways for differentiation.
                "title": (
                    p_data["title"].replace("â€™", "'") if p_data.get("title") else ""
                ),
                "selftext": (
                    p_data["selftext"].replace("â€™", "'")
                    if p_data.get("selftext")
                    else ""
                ),
                "upvote_ratio": p_data.get("upvote_ratio"),
                "ups": p_data.get("ups"),
                "downs": p_data.get("downs"),
                "score": p_data.get("score"),
                "edited": p_data.get("edited"),
                "all_awardings": p_data.get("all_awardings"),
                "controversiality": p_data.get("controversiality"),
                "num_comments": p_data.get("num_comments"),
                "link_flair_text": p_data.get("link_flair_text"),
                "created_utc": p_data.get("created_utc"),
            }
        )

# Convert collected posts to a DataFrame.
df = pd.DataFrame(all_posts)
df["created_time"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)
print("Combined Posts DataFrame:")
print(df.head())
print(df.shape)
print("\n" + "=" * 50 + "\n")

def extract_comments(comments_list, post_id, depth=0):
    """
    Recursively extract comment data from a list of comments.
    """
    extracted = []
    for comment in comments_list:
        # Skip non-comment objects (e.g., "more" objects).
        if comment["kind"] != "t1":
            continue
        data = comment["data"]
        extracted.append(
            {
                "post_id": post_id,
                "comment_id": data.get("id"),
                "parent_id": data.get("parent_id"),
                "author": data.get("author"),
                # Fix text encoding in comment body.
                "body": (
                    data.get("body").replace("â€™", "'") if data.get("body") else ""
                ),
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
        # Process nested replies if they exist.
        if data.get("replies") and isinstance(data["replies"], dict):
            child_comments = extract_comments(
                data["replies"]["data"]["children"], post_id, depth=depth + 1
            )
            extracted.extend(child_comments)
    return extracted

all_comments = []

# Process all posts (or a subset if desired) to extract comments.
print("Extracting comments from posts...")
for idx, row in tqdm(
    df.iterrows(), total=len(df), desc="Processing posts for comments"
):
    post_id = row["id"]
    comments_url = f"https://oauth.reddit.com/comments/{post_id}"
    r = requests.get(comments_url, headers=headers, params={"limit": 100})
    comments_json = r.json()
    # The second element in the response contains the comments listing.
    top_comments = comments_json[1]["data"]["children"]
    extracted_comments = extract_comments(top_comments, post_id)
    all_comments.extend(extracted_comments)

comments_df = pd.DataFrame(all_comments)
comments_df["created_time"] = pd.to_datetime(
    comments_df["created_utc"], unit="s", utc=True
).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
print("Combined Comments DataFrame:")
print(comments_df.head())
print("\n" + "=" * 50 + "\n")

# (Optional) Create separate DataFrames for each post if needed.
dfs_by_post = {}
for post_id in comments_df["post_id"].unique():
    dfs_by_post[post_id] = comments_df[comments_df["post_id"] == post_id]
for post_id, post_comments_df in dfs_by_post.items():
    print(f"Comments for post {post_id}:")
    print(post_comments_df)
    print("\n" + "-" * 40 + "\n")

# --------------------------------------------------
# Save the posts and comments DataFrames to CSV files.
# --------------------------------------------------
df.to_csv("reddit_posts_year.csv", index=False)
print("Saved posts to reddit_posts_year.csv")
comments_df.to_csv("reddit_comments_year.csv", index=False)
print("Saved comments to reddit_comments_year.csv")

# --------------------------------------------------
# Create a new CSV file with two columns:
#   - created_time: the UTC Zulu time (YYYY-MM-DDTHH:MM:SSZ)
#   - text: contains either the post title, post selftext, or comment body.
# For posts that have both a title and selftext, each becomes a separate row.
# --------------------------------------------------
text_rows = []

# Process posts: include title and selftext as separate data points.
for idx, row in df.iterrows():
    if row["title"].strip():
        text_rows.append({"Time Data": row["created_time"], "Headline": row["title"]})
    if row["selftext"].strip():
        text_rows.append(
            {"Time Data": row["created_time"], "Headline": row["selftext"]}
        )

# Process comments: use the comment body.
for idx, row in comments_df.iterrows():
    if row["body"].strip():
        text_rows.append({"Time Data": row["created_time"], "Headline": row["body"]})

text_df = pd.DataFrame(text_rows)
text_df.to_csv("reddit_texts_year.csv", index=False)
print("Saved combined texts to reddit_texts_year.csv")
