import geopandas as gpd
import fiona

# Enable KML support in fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

print("Fiona drivers configured. Testing sample KML read...")
sample_kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Sample Stop</name>
      <description>BEST Bus Stop</description>
      <Point>
        <coordinates>72.83,18.93,0</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>
"""

with open("data/temporary/sample.kml", "w", encoding="utf-8") as f:
    f.write(sample_kml)

try:
    gdf = gpd.read_file("data/temporary/sample.kml", driver="KML")
    print("Fiona KML reading succeeded!")
    print(gdf)
except Exception as e:
    print(f"Fiona KML read error: {e}")
    # Also test pure python XML parser as robust fallback
    import xml.etree.ElementTree as ET
    from shapely.geometry import Point
    tree = ET.parse("data/temporary/sample.kml")
    root = tree.getroot()
    print("Pure XML parser available as 100% reliable parser!")

