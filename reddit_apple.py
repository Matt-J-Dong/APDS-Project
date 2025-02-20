import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

client_id = 'lH3T4sATojFSEWZS5RBcEQ'
client_secret = 'bHPnWuBf8RhJz5YYnZaPokD1mIFMYw'
username = 'Somebody_Person'
password = os.environ.get("PASSWORD")
user_agent = 'Test/0.0.1'

# Set up HTTP Basic Auth with your client ID and secret
auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
print(auth)

data = {
    'grant_type': 'password',
    'username': username,
    'password': password
}

headers = {'User-Agent': user_agent}

# Request the access token
response = requests.post('https://www.reddit.com/api/v1/access_token',auth=auth, data=data, headers=headers)
#response = requests.post('http://httpbin.org/post',auth=auth, data=data, headers=headers)

print(response.json())

token = response.json()['access_token']

# Update headers with the access token
headers.update({'Authorization': f'bearer {token}'})

res = requests.get('https://oauth.reddit.com/r/apple/hot',headers=headers,params={'limit':100})
#print(res.json())
for post in res.json()['data']['children']:
    print(post['data']['title'])

rows = []
for post in res.json()['data']['children']: #Grabbing information from the Python subreddit
    rows.append({
        'subreddit': post['data']['subreddit'],
        'title': post['data']['title'],
        'selftext': post['data']['selftext'],
        'upvote_ratio': post['data']['upvote_ratio'],
        'ups': post['data']['ups'],
        'downs': post['data']['downs'],
        'score': post['data']['score']
    })

df = pd.DataFrame(rows)
print(df)