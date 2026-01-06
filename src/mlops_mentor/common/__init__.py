import os

from loguru import logger

GH_TOKEN = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
if not GH_TOKEN:
    logger.error("GitHub token not found in environment variables.")
    raise EnvironmentError("GitHub token not found in environment variables.")
github_headers = {"Authorization": f"Bearer {GH_TOKEN}"}
