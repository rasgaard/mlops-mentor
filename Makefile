scrape:
	uv run --env-file .env src/mlops_mentor/scraper/main.py scrape

formatting:
	uvx ruff check src/