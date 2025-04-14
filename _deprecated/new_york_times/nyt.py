import requests
from dotenv import load_dotenv
import os
import time
import csv
from datetime import datetime, timezone
import calendar

load_dotenv()

API_KEY = os.environ.get("NYT_API_KEY")
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
articles = []
start_year = 2024
start_month = 1
now = datetime.utcnow()
current_year = now.year
current_month = now.month

year = start_year
month = start_month
while (year < current_year) or (year == current_year and month <= current_month):
    begin_date = f"{year:04d}{month:02d}01"
    last_day = calendar.monthrange(year, month)[1]
    end_date = f"{year:04d}{month:02d}{last_day:02d}"

    params = {
        "fq": "headline:(Tesla)",  # Filter to only include articles with "Apple" in the headline
        "sort": "newest",  # Sort articles by newest first
        "page": 0,  # Get the first page (up to 10 articles)
        "api-key": API_KEY,
        "begin_date": begin_date,
        "end_date": end_date,
    }

    print(
        f"Fetching articles for {year}-{month:02d} (from {begin_date} to {end_date})..."
    )
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        docs = data.get("response", {}).get("docs", [])
        articles.extend(docs)
    else:
        print(f"Error fetching articles for {year}-{month:02d}: {response.status_code}")

    if month == 12:
        month = 1
        year += 1
    else:
        month += 1

    if (year < current_year) or (year == current_year and month <= current_month):
        time.sleep(12)

print(f"Retrieved {len(articles)} articles.")

csv_rows = []
for article in articles:
    pub_date_str = article.get("pub_date")
    if pub_date_str:
        try:
            dt = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%S%z")
            time_data = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            print(f"Error parsing date {pub_date_str}: {e}")
            time_data = "Invalid Date"
    else:
        time_data = "No Date"
    headline = article.get("headline", {}).get("main", "No Headline")
    csv_rows.append({"Time Data": time_data, "Headline": headline})
csv_file = "nyt_tesla_articles.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Time Data", "Headline"])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"Data written to {csv_file}.")
