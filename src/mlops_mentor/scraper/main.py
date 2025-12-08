import datetime
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import requests
from datasets import Dataset
from loguru import logger
from typer import Typer

from mlops_mentor.common import github_headers as headers
from mlops_mentor.common.data import load_groups
from mlops_mentor.scraper.models import RepoContent, Report, RepoStats


def create_activity_matrix(
    commits: list,
    max_delta: int = 5,
    min_delta: int = 1,
) -> list[list[int]]:
    """Creates an activity matrix from the commits."""
    commit_times = [
        datetime.datetime.fromisoformat(commit["commit"]["committer"]["date"][:-1])
        for commit in commits
    ]
    commit_times.sort()

    start_time = commit_times[0]
    end_time = max(
        start_time + datetime.timedelta(weeks=min_delta),
        min(start_time + datetime.timedelta(weeks=max_delta), commit_times[-1]),
    )

    num_days = (end_time - start_time).days + 1  # include last day

    commit_matrix = np.zeros((num_days, 24), dtype=int)

    for commit_time in commit_times:
        if start_time <= commit_time <= end_time:
            day_index = (commit_time - start_time).days
            hour_index = commit_time.hour
            commit_matrix[day_index, hour_index] += 1

    return commit_matrix.tolist()


app = Typer()


@app.command()
def scrape(push_to_hub: bool = False, hub_repo_id: str = "your-username/repo-stats"):
    """Main function to scrape the group-repository data."""
    logger.info("Getting group-repository information")
    # download_data("group_info.csv")
    groups = load_groups("group_info.csv")
    logger.info("Group-repository information loaded successfully")

    repo_stats: list[RepoStats] = []
    for index, group in enumerate(groups):
        logger.info(
            f"Processing group {group.group_number}, {index + 1}/{len(groups)}. Accessible: {group.repo_accessible}"
        )
        if group.repo_accessible:
            contributors = group.contributors
            num_contributors = len(contributors)

            prs = group.prs
            num_prs = len(prs)

            commits = group.commits
            num_commits_to_main = len(commits)
            commit_messages = [c["commit"]["message"] for c in commits]
            average_commit_length_to_main = sum(
                [len(c) for c in commit_messages]
            ) / len(commit_messages)
            latest_commit = commits[0]["commit"]["author"]["date"]

            merged_prs = [p["number"] for p in prs if p["merged_at"] is not None]
            for pr_num in merged_prs:
                pr_commits: list[dict] = requests.get(
                    f"{group.repo_api}/pulls/{pr_num}/commits",
                    headers=headers,
                    timeout=100,
                ).json()
                commit_messages += [c["commit"]["message"] for c in pr_commits]
                for commit in pr_commits:
                    for contributor in contributors:
                        commit_author = commit.get("author")  # GitHub account info
                        commit_committer = commit.get(
                            "committer"
                        )  # GitHub account info
                        commit_author_name = commit["commit"]["author"]["name"]
                        commit_committer_name = commit["commit"]["committer"]["name"]

                        matches = (
                            (
                                commit_author
                                and commit_author.get("login") == contributor.login
                            )
                            or (
                                commit_author_name
                                and commit_author_name.lower()
                                == contributor.login.lower()
                            )
                            or (
                                commit_committer
                                and commit_committer.get("login") == contributor.login
                            )
                            or (
                                commit_committer_name
                                and commit_committer_name.lower()
                                == contributor.login.lower()
                            )
                        )

                        if matches:
                            contributor.commits_pr += 1
                            break
                commits += pr_commits

            activity_matrix = create_activity_matrix(commits, max_delta=3, min_delta=1)

            average_commit_length = sum([len(c) for c in commit_messages]) / len(
                commit_messages
            )

            contributions_per_contributor = [c.total_commits for c in contributors]
            total_commits = sum(contributions_per_contributor)

            repo_content = RepoContent(
                group_number=group.group_number,
                repo_api=group.repo_api,
                default_branch=group.default_branch,
            )
            num_docker_files = repo_content.num_docker_files
            num_python_files = repo_content.num_python_files
            num_workflow_files = repo_content.num_workflow_files
            has_requirements_file = repo_content.has_requirements_file
            has_cloudbuild = repo_content.has_cloudbuild
            using_dvc = repo_content.using_dvc
            repo_size = repo_content.repo_size
            readme_length = repo_content.readme_length
            actions_passing = repo_content.actions_passing

            report = Report(
                group_number=group.group_number,
                repo_api=group.repo_api,
                default_branch=group.default_branch,
            )
            num_warnings = report.check_answers
        else:
            num_contributors = None
            num_prs = None
            num_commits_to_main = None
            average_commit_length_to_main = None
            latest_commit = None
            average_commit_length = None
            total_commits = None
            contributions_per_contributor = None
            total_commits = None
            activity_matrix = None

            num_docker_files = None
            num_python_files = None
            num_workflow_files = None
            has_requirements_file = None
            has_cloudbuild = None
            using_dvc = None
            repo_size = None
            readme_length = None
            actions_passing = None

            num_warnings = None

        repo_stat = RepoStats(
            group_number=group.group_number,
            group_size=group.group_size,
            num_contributors=num_contributors,
            num_prs=num_prs,
            num_commits_to_main=num_commits_to_main,
            average_commit_length_to_main=average_commit_length_to_main,
            latest_commit=latest_commit,
            average_commit_length=average_commit_length,
            contributions_per_contributor=contributions_per_contributor,
            total_commits=total_commits,
            activity_matrix=activity_matrix,
            num_docker_files=num_docker_files,
            num_python_files=num_python_files,
            num_workflow_files=num_workflow_files,
            has_requirements_file=has_requirements_file,
            has_cloudbuild=has_cloudbuild,
            using_dvc=using_dvc,
            repo_size=repo_size,
            readme_length=readme_length,
            actions_passing=actions_passing,
            num_warnings=num_warnings,
        )
        repo_stats.append(repo_stat)

    logger.info("Writing repo stats to file")
    now = datetime.datetime.now(tz=datetime.UTC).strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"repo_stats_{now}.json"

    # Convert to list of dicts for serialization
    repo_stats_dicts = [r.model_dump() for r in repo_stats]

    with open("repo_stats.json", "w") as f:
        json.dump(repo_stats_dicts, f)
    with open(filename, "w") as f:
        json.dump(repo_stats_dicts, f)

    # Create and optionally push Hugging Face dataset
    if push_to_hub:
        logger.info("Creating Hugging Face dataset")
        dataset = Dataset.from_list(repo_stats_dicts)

        logger.info(f"Pushing dataset to Hugging Face Hub: {hub_repo_id}")
        dataset.push_to_hub(
            hub_repo_id, private=True
        )  # Set private=False if you want it public
        logger.info("Dataset successfully pushed to Hugging Face Hub")
    else:
        logger.info("Skipping Hugging Face Hub upload (use --push-to-hub to enable)")

    logger.info("Cleaning locally temp files")
    Path("README.md").unlink()
    Path("report.py").unlink()
    Path(filename).unlink()


@app.command()
def clone(base_dir: str = "cloned_repos"):
    """Clones the repositories of the groups."""
    logger.info("Getting group-repository information")
    groups = load_groups()
    logger.info("Group-repository information loaded successfully")

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for group in groups:
        repo_url = group.repo_url
        group_number = group.group_number

        # Create a directory for the group if it doesn't exist
        group_dir = os.path.join(base_dir, f"group_{group_number}")
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        # Extract the repository name from the URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")

        # Create a directory for the repository
        repo_dir = os.path.join(group_dir, repo_name)
        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir)

        # Clone the repository
        try:
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
            logger.info(f"Successfully cloned {repo_url} into {repo_dir}")
        except subprocess.CalledProcessError as e:
            logger.info(f"Failed to clone {repo_url}: {e}")


if __name__ == "__main__":
    app()
