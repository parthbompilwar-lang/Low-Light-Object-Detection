import urllib.request
import re

try:
    resp = urllib.request.urlopen("http://localhost:8501")
    html = resp.read().decode("utf-8")
    print("HTTP status:", resp.status)
    scripts = re.findall(r'src="([^"]+)"', html)
    print("Scripts found:", scripts)
    for s in scripts:
        url = "http://localhost:8501" + s if s.startswith("/") else s
        try:
            r = urllib.request.urlopen(url)
            print(f"OK: {url} -> status {r.status}, size {len(r.read())}")
        except Exception as e:
            print(f"FAILED: {url} -> {e}")
except Exception as e:
    print("Root error:", e)
