import requests
import json
import urllib3
urllib3.disable_warnings()

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

print("=== Search OpenCity for Suburban / Railway ===")
for q in ['suburban', 'mumbai local', 'mumbai transit', 'mumbai railway', 'station']:
    try:
        r = requests.get(f'https://data.opencity.in/api/3/action/package_search?q={q}', timeout=10, headers=headers)
        if r.status_code == 200:
            pkgs = r.json().get('result', {}).get('results', [])
            print(f"Query '{q}' found {len(pkgs)} packages:")
            for p in pkgs[:3]:
                print(f"   * {p.get('name')}: {p.get('title')}")
                for res in p.get('resources', []):
                    if any(ext in res.get('url', '').lower() for ext in ['.kml', '.geojson', '.csv', '.shp', '.zip']):
                        print(f"       Resource: {res.get('name')} | {res.get('format')} | {res.get('url')}")
    except Exception as e:
        print(f"Query '{q}' failed: {e}")

print("\n=== WorldPop 2020 India Raster Info ===")
try:
    r = requests.get('https://hub.worldpop.org/rest/data/pop/wpgp?iso3=IND', timeout=10, headers=headers)
    if r.status_code == 200:
        items = r.json().get('data', [])
        # Find 2020 or latest
        latest = [item for item in items if item.get('popyear') == '2020']
        if latest:
            item = latest[0]
            print(f"Found latest 2020 product: {item.get('title')}")
            files = item.get('files', [])
            print(f"Files: {files}")
            if files:
                head = requests.head(files[0], timeout=10, headers=headers)
                size_mb = int(head.headers.get('content-length', 0)) / (1024*1024)
                print(f"Direct download URL: {files[0]} | Size: {size_mb:.2f} MB")
except Exception as e:
    print(f"WorldPop check failed: {e}")
