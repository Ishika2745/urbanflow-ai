import requests
import re

r = requests.get('https://api.worldpop.org/v3/assets/index-meeBE9d5.js')
text = r.text
matches = re.findall(r'https?://api\.worldpop\.org/v3/[a-zA-Z0-9_\-\/\?&=]+', text)
print("Explicit v3 URLs:", set(matches))

# Also search for endpoint examples (e.g. /v3/...)
examples = re.findall(r'/v3/[a-zA-Z0-9_\-\/\?&=]+', text)
print("Relative v3 endpoints:", set(examples))

