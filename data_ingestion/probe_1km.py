import requests
import re

for url in [
    'https://data.worldpop.org/GIS/Population/Individual_countries/IND/',
    'https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IND/'
]:
    try:
        r = requests.get(url, timeout=10)
        print(f"{url} -> Status {r.status_code}")
        if r.status_code == 200:
            links = re.findall(r'href="([^"]+)"', r.text)
            print("  Links:", [l for l in links if not l.startswith('?')][:10])
    except Exception as e:
        print(f"{url} -> Error: {e}")

