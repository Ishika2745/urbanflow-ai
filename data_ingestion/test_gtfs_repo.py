import requests
import zipfile
import io

url = 'https://github.com/croyla/mumbai-gtfs/archive/refs/heads/master.zip'
print("Checking croyla/mumbai-gtfs zip content...")
r = requests.get(url, stream=True, timeout=30)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("Files in archive:")
    for name in z.namelist()[:25]:
        print("  -", name)

