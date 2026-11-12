import requests
import urllib3
urllib3.disable_warnings()

headers = {'User-Agent': 'Mozilla/5.0'}

targets = {
    'best_kml': 'https://data.opencity.in/dataset/c2a2de1e-a017-4fb9-9e46-de6fbfbb393b/resource/514eadc2-7fa8-4749-87e5-ebe486487f09/download/b34111da-5b73-45e9-82d0-78da9d153b1c.kml',
    'railway_stations_kml': 'https://data.opencity.in/dataset/50851056-1954-459f-a747-09366dcd416d/resource/142fbd25-9ceb-41b2-900d-72479cdc179e/download/5baa7e83-2855-4d17-86a9-9954b044ec08.kml',
    'wards_kml': 'https://data.opencity.in/dataset/60f27371-717e-40b6-8554-fdc7e6c10437/resource/0318c3e8-1530-4bf4-b29b-7281573dee8a/download/e7a671e2-1f71-4219-a83c-556334bc9021.kml'
}

for name, url in targets.items():
    try:
        r = requests.get(url, timeout=15, headers=headers, verify=False)
        print(f"{name}: status={r.status_code}, length={len(r.content)} bytes, starts_with={r.text[:60]}")
    except Exception as e:
        print(f"{name}: error {e}")

