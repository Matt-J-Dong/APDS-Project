import requests

API_KEY = "QzAXGxtDOmWxeHSclpvkqt8F8VB2Htf7"
url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
params = {
    "q": "Apple",
    "api-key": API_KEY
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    articles = data.get("response", {}).get("docs", [])
    
    # Print details for each article
    for article in articles:
        headline = article.get("headline", {}).get("main", "No headline")
        web_url = article.get("web_url", "No URL")
        pub_date = article.get("pub_date", "No publication date")
        print(f"Headline: {headline}")
        print(f"URL: {web_url}")
        print(f"Published Date: {pub_date}")
        print("-" * 40)
else:
    print("ERROR, SOMETHING WENT WRONG:", response.status_code)
