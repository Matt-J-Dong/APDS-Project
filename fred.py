import requests

API_KEY = "aba251875bb53e4ed3c2b74e30f19ace"
series_id = "INTDSRUSM193N" #Any indicator found on FRED

url = "https://api.stlouisfed.org/fred/series/observations"

# Define parameters for the API call
params = {
    "series_id": series_id,
    "api_key": API_KEY,
    "file_type": "json",
    # Optional: limit to a specific time range
    "observation_start": "2010-01-01",
    "observation_end": "2020-12-31"
}

response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    observations = data.get("observations", [])
    for obs in observations[:10]:
        date = obs.get("date", "N/A")
        value = obs.get("value", "N/A")
        print(f"Date: {date}, Value: {value}")
else:
    print("ERROR, SOMETHING WENT WRONG", response.status_code, response.text)
