import os
import requests

TOKEN = os.environ["lichess_token"]
H = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/x-ndjson",
}

# Smoke Test for
def smoke_test_token():
    global TOKEN
    H = {"Authorization": f"Bearer {TOKEN}"}
    me = requests.get("https://lichess.org/api/account", headers=H).json()
    print("Logged in as:", me["username"])

smoke_test_token()