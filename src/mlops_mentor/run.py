import time

from datasets import Dataset
from loguru import logger

from mlops_mentor.common.data import load_groups
from mlops_mentor.llm_judge import codebase
from mlops_mentor.scraper import scrape

if __name__ == "__main__":
    groups = load_groups("group_info.csv")
    results = []
    for group in groups:
        if not group.repo_info.is_accessible:
            logger.warning(
                f"Skipping inaccessible repository: {group.repo_info.repo_url}"
            )
            continue

        repo_url = group.repo_info.repo_url
        group_results = {
            "timestamp": time.time(),
            "group_number": group.group_number,
            "repo_url": repo_url,
        }
        print(f"Scraping repository: {repo_url}")
        stats = scrape(repo_url)
        print(f"Repository stats: {stats}")
        group_results["stats"] = stats.model_dump_json()

        print(f"Evaluating codebase for repository: {repo_url}")
        evaluation = codebase(repo_url)
        print(f"Codebase evaluation: {evaluation}")
        group_results["evaluation"] = evaluation.model_dump_json()

        results.append(group_results)

    dataset = Dataset.from_list(results)
    dataset.push_to_hub("rasgaard/repo-evaluations", private=True)
