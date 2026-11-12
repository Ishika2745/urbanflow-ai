import requests
import json

r = requests.get('https://hub.worldpop.org/rest/data/pop/wpgp?iso3=IND', timeout=15)
if r.status_code == 200:
    for d in r.json().get('data', []):
        year = d.get('popyear')
        title = d.get('title')
        files = d.get('files', [])
        print(f"Year {year}: {title}")
        for f in files:
            print("   ->", f)

