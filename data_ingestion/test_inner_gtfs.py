import requests
import zipfile
import io

url = 'https://github.com/croyla/mumbai-gtfs/archive/refs/heads/master.zip'
r = requests.get(url, stream=True, timeout=30)
z = zipfile.ZipFile(io.BytesIO(r.content))

for compat_name in ['mumbai-gtfs-main/gtfs_compat.zip', 'mumbai-gtfs-master/gtfs_compat.zip']:
    if compat_name in z.namelist():
        print(f"Extracting {compat_name}...")
        inner_zip_bytes = z.read(compat_name)
        inner_z = zipfile.ZipFile(io.BytesIO(inner_zip_bytes))
        print("Inner files:")
        for fn in inner_z.namelist():
            print("  *", fn)

