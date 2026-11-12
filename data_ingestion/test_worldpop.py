import requests

url = 'https://hub.worldpop.org/rest/data/pop/wpgp?iso3=IND'
r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
if r.status_code == 200:
    data = r.json().get('data', [])
    print(f"Total datasets for IND: {len(data)}")
    for d in data:
        popyear = d.get('popyear')
        title = d.get('title')
        id_val = d.get('id')
        print(f"ID: {id_val} | Year: {popyear} | Title: {title}")
        desc = d.get('desc')
        # Check files
        files = d.get('files', [])
        if files:
            print(f"   Sample file: {files[0]}")

