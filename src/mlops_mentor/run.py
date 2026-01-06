from loguru import logger

from mlops_mentor.common.data import load_groups
from mlops_mentor.llm_judge import codebase
from mlops_mentor.scraper import scrape

if __name__ == "__main__":
    groups = load_groups("group_info.csv")
    for group in groups:
        if not group.repo_info.is_accessible:
            logger.warning(
                f"Skipping inaccessible repository: {group.repo_info.repo_url}"
            )
            continue
        repo_url = group.repo_info.repo_url
        group.repo_info
        print(f"Scraping repository: {repo_url}")
        stats = scrape(repo_url)
        print(f"Repository stats: {stats}")

        print(f"Evaluating codebase for repository: {repo_url}")
        evaluation = codebase(repo_url)
        print(f"Codebase evaluation: {evaluation}")

    # TODO: Save results to Hugging Face dataset
