import json
import os
import shutil
from pathlib import Path
from pprint import pprint

import typer
from loguru import logger
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider

from mlops_mentor.common.data import load_groups
from mlops_mentor.llm_judge.models import (
    RepoMix,
    TACodeResponse,
    TADependency,
    TAReportResponse,
)
from mlops_mentor.llm_judge.utils import get_repo_content

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

model = OpenAIChatModel(
    "gpt-oss",
    provider=LiteLLMProvider(
        api_base="https://chat.campusai.compute.dtu.dk/api/v1",
        api_key=os.getenv("CAMPUSAI_API_KEY"),
    ),
)


def finalize(responses: list, clean: bool = True, name: str = "responses.json") -> None:
    """Save responses and clean up if needed."""
    with open(name, "w") as f:  # Save responses in case of error
        json.dump([response.model_dump() for response in responses], f, indent=4)
    if clean:
        shutil.rmtree(Path("output"))


@app.command()
def codebase(group_nb: None | int = None, clean: bool = True) -> None:
    """Main function to evaluate the codebase of a group."""
    ta_agent = Agent(
        model=model,
        deps_type=TADependency,
        output_type=TACodeResponse,  # Changed from output_type
        system_prompt="""You are a teaching assistant for a university level course on machine learning operations.
        Your job is to review and evaluate the students final group project.

        You must provide scores as integers in the specified ranges:
        
        CODE QUALITY (1-5):
        1: Poor - Major violations of best practices, unreadable
        2: Below Average - Significant readability/maintainability issues
        3: Average - Meets basic standards, room for improvement
        4: Good - Follows most best practices, minor issues only
        5: Excellent - Clean, maintainable, follows all best practices

        UNIT TESTING (1-5):
        1: Poor - No or minimal tests, negligible coverage
        2: Below Average - Tests exist but inadequate coverage
        3: Average - Adequate coverage with noticeable gaps
        4: Good - Good coverage of critical functionality
        5: Excellent - Comprehensive test coverage

        CI/CD (1-5):
        1: Poor - No pipeline or completely broken
        2: Below Average - Exists but unreliable or manual
        3: Average - Functional but missing best practices
        4: Good - Reliable and mostly automated
        5: Excellent - Robust, fully automated, follows best practices

        SUMMARY: Provide 2-3 paragraphs (max 500 words) covering code quality, testing, CI/CD, and improvement suggestions.

        OVERALL SCORE (1-10): Rate the entire MLOps pipeline implementation.
        CONFIDENCE (1-10): Your confidence in the assessment.

        Example valid response:
        {
            "code_quality": 4,
            "unit_testing": 3,
            "ci_cd": 2,
            "summary": "The codebase demonstrates good structure and readability...",
            "overall_score": 6,
            "confidence": 8
        }
        """,
    )

    @ta_agent.system_prompt
    async def add_group_information(ctx: RunContext[TADependency]) -> str:
        group_number = ctx.deps.group_info.group_number
        repo_content = get_repo_content(ctx.deps.group_info.repo_url, ctx.deps.repomix)
        return f"""
        Group {group_number} repository:
        
        {repo_content}
        
        Analyze this repository and provide scores following the exact JSON format specified.
        """

    group_data = load_groups()
    if group_nb:
        group_data = [group_data[group_nb - 1]]

    responses: list[TACodeResponse] = []
    for group in group_data:
        if not group.repo_accessible:
            logger.warning(
                f"Skipping group {group.group_number} as the repository is not accessible"
            )
            continue
        logger.info(f"Processing group {group.group_number}")
        deps = TADependency(
            group_info=group,
            repomix=RepoMix(
                ignore=RepoMix.Ignore(
                    customPatterns=[
                        ".dvc/*",
                        "*.dvc",
                        "**/*.js",
                        "**/*.css",
                        "**/*.map",
                        "**/*.svg",
                        "**/*/report.py",
                        "reports/README.md",
                        "*.gitignore",
                        "**/*.ipynb",
                        "**/*.html",
                        "uv.lock",
                        "data/**",
                        "**/*.csv",
                        "log/**",
                        "logs/**",
                        "outputs/**",
                    ]
                )
            ),
        )
        try:
            logger.debug(f"System prompt : {ta_agent._system_prompts}")
            result = ta_agent.run_sync(
                "Evaluate this group's repository and return a JSON response with all required fields.",
                deps=deps,
            )
            #            result.output.request_usage = result.usage()  # Changed from result.output
            pprint(result.output)
            responses.append(result.output)
        except Exception as e:
            logger.error(f"Failed for group {group.group_number}: {e}")
            finalize(responses, clean, name="codebase")
            raise e
    finalize(responses, clean, name="codebase")


@app.command()
def report(group_nb: None | int = None, clean: bool = True) -> None:
    """Main function to evaluate the report of a group."""
    ta_agent = Agent(
        model=model,
        deps_type=TADependency,
        output_type=TAReportResponse,
        system_prompt="""
        You are a teaching assistant for a university level course on machine learning operations. You are tasked with
        correcting a student's report which is provided in markdown format. The report is a template consisting of 31
        questions and are fologrmatted into a couple of sections:  Group information, Coding environment, Version
        control, Running code and tracking experiments, Working in the cloud, Deployment, Overall discussion of project.
        Additionally, it contains a checklist of 52 items that needs to be filled out. For each of the sections
        (except Group information), you will provide a brief summary of the student's response and then provide feedback
        on the accuracy and completeness of the response. You will also provide suggestions for improvement. Score each
        section on a scale from 1 to 5 based on the following criteria:
        1: Poor - The response is inaccurate, incomplete, or contains significant errors.
        2: Below Average - The response is partially accurate but contains several errors or omissions.
        3: Average - The response is mostly accurate but contains minor errors or omissions.
        4: Good - The response is accurate and complete with only minor issues.
        5: Excellent - The response is accurate, complete, and well-reasoned.
        You should penelise the students for not answering questions. Only focus on the students answers. In addition
        you need to return how many of the 52 items from the checklist were completed. Provide also a summary of the
        overall report evaluation using no more than 300 words. Finally, return a grading score from 1-10 and your
        confidence in the grading from 1-10.
        """,
    )

    @ta_agent.system_prompt
    async def add_group_information(ctx: RunContext[TADependency]) -> str:
        group_number = ctx.deps.group_info.group_number
        repo_content = get_repo_content(ctx.deps.group_info.repo_url, ctx.deps.repomix)
        return f"""
        Group {group_number} has submitted the following report:
        {repo_content}
        """

    group_data = load_groups()
    if group_nb:
        group_data = [group_data[group_nb - 1]]

    responses: list[TAReportResponse] = []
    for group in group_data:
        deps = TADependency(
            group_info=group, repomix=RepoMix(include=["reports/README.md"])
        )
        try:
            result = ta_agent.run_sync(
                "What do you think of the groups report?", deps=deps
            )
            result.output.request_usage = result.usage()
            pprint(result.output)
            responses.append(result.output)
        except Exception as e:
            finalize(responses, clean, name="codebase")
            raise e
    finalize(responses, clean, name="codebase")


if __name__ == "__main__":
    app()
