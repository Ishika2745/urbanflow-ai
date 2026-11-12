import requests

# Test worldpop API endpoints
endpoints = [
    'https://api.worldpop.org/v3/documentation',
    'https://api.worldpop.org/v3/tasks',
    'https://api.worldpop.org/v3/stats',
    'https://hub.worldpop.org/rest/data/pop/wpgp?iso3=IND'
]

for ep in endpoints:
    try:
        r = requests.get(ep, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        print(f"{ep} -> Status: {r.status_code}, Length: {len(r.content)}")
        if r.status_code == 200 and 'json' in r.headers.get('content-type', ''):
            print("  JSON keys:", list(r.json().keys())[:5])
    except Exception as e:
        print(f"{ep} -> Error: {e}")

