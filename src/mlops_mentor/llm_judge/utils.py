import os
from pathlib import Path

from loguru import logger

from mlops_mentor.llm_judge.models import RepoMix


def call_repomix(
    repo: str, repomix_config: RepoMix, out_folder: str = "output"
) -> None:
    """Call repomix on a repository."""
    repomix_config.dump_json("repomix.config.json")
    logger.info(f"Running repomix on {repo}")
    current_dir = os.getcwd()
    os.system(
        f"repomix -c {current_dir}/repomix.config.json --remote {repo} --verbose >> output.log"
    )
    repo_name = "_".join(repo.split("/")[-2:])
    os.system(f"mkdir -p  {out_folder}/{repo_name}")
    os.system(f"mv output.log {out_folder}/{repo_name}/output.log")
    os.system(f"mv repomix-output.md {out_folder}/{repo_name}/repomix-output.md")
    os.system("rm repomix.config.json")


def get_repo_content(repository: str, repomix_config: RepoMix) -> str:
    """Get the code from a repository."""
    if repository.startswith("https://github.com"):
        call_repomix(repository, repomix_config, out_folder="output")
        repo_name = "_".join(repository.split("/")[-2:])
        path = Path(f"output/{repo_name}/repomix-output.md")
    else:
        path = Path(repository)
    with path.open("r") as file:
        return file.read()
