import requests
import re

r = requests.get('https://api.worldpop.org/v3/documentation')
js_files = re.findall(r'src="([^"]+\.js)"', r.text)
print("JS files:", js_files)
for js in js_files:
    r_js = requests.get('https://api.worldpop.org' + js)
    endpoints = set(re.findall(r'/(?:v3|stats|pop)/[a-zA-Z0-9_\-\/]+', r_js.text))
    print("Found API endpoints in JS:")
    for e in sorted(endpoints):
        if len(e) < 50:
            print("  ", e)

