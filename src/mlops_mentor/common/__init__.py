import os

GH_TOKEN = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
github_headers = {"Authorization": f"Bearer {GH_TOKEN}"}
