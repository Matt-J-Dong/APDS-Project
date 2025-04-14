import requests
from dotenv import load_dotenv
import os
import time
import csv
import json
import calendar
from datetime import datetime, timezone

load_dotenv()

# Configuration and API key
API_KEY = os.environ.get("NYT_API_KEY")
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# Define the keyword groups for the companies.
keyword_groups = {
    "apple": ["apple", "aapl"],
    "microsoft": ["microsoft", "msft"],
    "nvidia": ["nvidia", "nvda"],
    "amazon": ["amazon", "amzn"],
    "google": ["alphabet", "google", "goog", "googl"],
    "meta": ["meta", "facebook", "meta platforms", "meta platforms (facebook)"],
    "tesla": ["tesla", "tsla"],
    "broadcom": ["broadcom", "avgo"],
    "netflix": ["netflix", "nflx"],
    "oracle": ["oracle", "orcl"],
    "salesforce": ["salesforce", "crm"],
    "cisco": ["cisco", "csco"],
    "ibm": ["ibm"],
    "palantir": ["palantir", "pltr"],
    "intuit": ["intuit", "intu"],
    "servicenow": ["servicenow", "now"],
    "adobe": ["adobe", "adbe"],
    "qualcomm": ["qualcomm", "qcom"],
    "amd": ["amd"],
    "texas_instruments": ["texas instruments", "txn"],
    "uber": ["uber"],
    "booking": ["booking", "bkng", "booking.com"],
    "adp": ["automatic data processing", "adp"],
    "fiserv": ["fiserv", "fi"],
    "applied_materials": ["applied materials", "amat"],
    "palo_alto": ["palo alto networks", "panw"],
    "intel": ["intel", "intc"],
}

# Files for progress and output data.
PROGRESS_FILE = "progress.json"
CSV_FILE = "nyt_articles.csv"

# Start date is January 2015.
START_YEAR = 2015
START_MONTH = 1

# Get the current UTC date.
now = datetime.utcnow()
CURRENT_YEAR = now.year
CURRENT_MONTH = now.month

# Function to compare (year, month) tuples.
def is_before_or_equal(year, month, ref_year, ref_month):
    return (year < ref_year) or (year == ref_year and month <= ref_month)

# Function to get the next month given a year and month.
def next_year_month(year, month):
    if month == 12:
        return year + 1, 1
    else:
        return year, month + 1

# Load or initialize the progress
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as pf:
        progress = json.load(pf)
else:
    # Initialize progress so that for each company we start at 2015-01.
    progress = {company: {"year": START_YEAR, "month": START_MONTH} for company in keyword_groups}

# Prepare to store articles.
all_articles = []

# Initialize API request count
requests_count = 0
DAILY_LIMIT = 500

# Iterate over companies in sorted order (ensuring consistent order across runs)
for company in sorted(keyword_groups.keys()):
    # Check if we already completed processing for this company:
    comp_prog = progress.get(company, {"year": START_YEAR, "month": START_MONTH})
    prog_year = comp_prog["year"]
    prog_month = comp_prog["month"]
    
    # Continue with months up to the current month in UTC.
    while is_before_or_equal(prog_year, prog_month, CURRENT_YEAR, CURRENT_MONTH):
        # Check if the daily limit has been reached.
        if requests_count >= DAILY_LIMIT:
            print("Reached the daily limit of 500 requests. Saving progress and stopping for today.")
            # Save progress to file.
            with open(PROGRESS_FILE, "w") as pf:
                json.dump(progress, pf, indent=4)
            # Exit the loops early.
            break

        # Construct the date range for the current month.
        begin_date = f"{prog_year:04d}{prog_month:02d}01"
        last_day = calendar.monthrange(prog_year, prog_month)[1]
        end_date = f"{prog_year:04d}{prog_month:02d}{last_day:02d}"
        
        # Construct the NYT API query. Join the keywords with " OR " inside the headline filter.
        keywords_list = keyword_groups[company]
        query_string = f"headline:({' OR '.join(keywords_list)})"
        
        # Define parameters for the API request.
        params = {
            "fq": query_string,
            "sort": "newest",
            "page": 0,
            "api-key": API_KEY,
            "begin_date": begin_date,
            "end_date": end_date,
        }
        
        print(
            f"Fetching articles for '{company}' for {prog_year}-{prog_month:02d} (from {begin_date} to {end_date})..."
        )
        
        try:
            response = requests.get(BASE_URL, params=params)
        except Exception as e:
            print(f"Request error: {e}")
            response = None
        
        requests_count += 1  # Count this request

        if response is not None and response.status_code == 200:
            data = response.json()
            # Use "or []" to ensure docs is an empty list if None
            docs = data.get("response", {}).get("docs") or []
            # Add a 'Company' field to each article and store them.
            for article in docs:
                article["Company"] = company
            all_articles.extend(docs)
        else:
            status = response.status_code if response is not None else "No response"
            print(f"Error fetching articles for {company} on {prog_year}-{prog_month:02d}: {status}")
        
        # Update the progress for this company (i.e. move to the next month).
        prog_year, prog_month = next_year_month(prog_year, prog_month)
        progress[company]["year"] = prog_year
        progress[company]["month"] = prog_month

        # Sleep for 12 seconds between requests to adhere to rate limits.
        time.sleep(12)
    
    # If we reached the daily request limit, break out of the outer loop.
    if requests_count >= DAILY_LIMIT:
        break

# Save progress (so that the next run resumes at the right position).
with open(PROGRESS_FILE, "w") as pf:
    json.dump(progress, pf, indent=4)

print(f"Total API requests made in this run: {requests_count}")
print(f"Total articles retrieved in this run: {len(all_articles)}")

# Prepare to write (or append) the output CSV.
# We'll store three columns: Company, Time Data, and Headline.
csv_fieldnames = ["Company", "Time Data", "Headline"]

# Prepare CSV rows by formatting publication date into UTC ISO format.
csv_rows = []
for article in all_articles:
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
    company = article.get("Company", "Unknown")
    csv_rows.append({"Company": company, "Time Data": time_data, "Headline": headline})

# Check if the CSV file already exists. Append if it does, otherwise write a new file with header.
file_exists = os.path.exists(CSV_FILE)
with open(CSV_FILE, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerows(csv_rows)

print(f"Data written to {CSV_FILE}.")
