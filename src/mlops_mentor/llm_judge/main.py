import json
import os
import shutil
from pathlib import Path
from pprint import pprint

from loguru import logger
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.providers.ollama import OllamaProvider

from mlops_mentor.llm_judge.models import (
    CICDResponse,
    CodeQualityResponse,
    RepoMix,
    TACodeResponse,
    TADependency,
    TAReportResponse,
    UnitTestingResponse,
)
from mlops_mentor.llm_judge.utils import get_repo_content

ollama_provider = OllamaProvider(base_url="http://localhost:11434/v1")
litellm_provider = LiteLLMProvider(
    api_base="https://chat.campusai.compute.dtu.dk/api/v1",
    api_key=os.getenv("CAMPUSAI_API_KEY"),
)

model = OpenAIChatModel(
    model_name="ministral-3:8b-32k",
    provider=ollama_provider,
)


def finalize(responses: list, clean: bool = True, name: str = "responses.json") -> None:
    """Save responses and clean up if needed."""
    with open(name, "w") as f:  # Save responses in case of error
        json.dump([response.model_dump() for response in responses], f, indent=4)
    if clean:
        shutil.rmtree(Path("output"))


def repo_context(ctx: RunContext[TADependency], context_type: str = "code") -> str:
    repo_content = get_repo_content(ctx.deps.repo_link, ctx.deps.repomix)
    return f"{context_type}:\n\n{repo_content}"


def create_code_quality_agent() -> Agent[TADependency, CodeQualityResponse]:
    """Create an agent focused on code quality evaluation."""
    agent = Agent(
        model=model,
        deps_type=TADependency,
        output_type=CodeQualityResponse,
        system_prompt="""You are a teaching assistant evaluating code quality for a university MLOps course.

Evaluate the code based on:
- Code structure and organization
- Adherence to Python best practices (PEP 8, type hints, docstrings)
- Readability and maintainability
- Proper use of design patterns
- Configuration management
- Documentation quality

SCORE (1-5):
1: Poor - Major violations, unreadable code
2: Below Average - Significant issues with readability/maintainability
3: Average - Meets basic standards, room for improvement
4: Good - Follows most best practices, minor issues only
5: Excellent - Clean, maintainable, follows all best practices

SUMMARY: 1-2 paragraphs (max 200 words) on code quality findings and suggestions.
CONFIDENCE (1-10): Your confidence in the assessment.
""",
    )

    @agent.system_prompt
    async def add_repo_context(ctx: RunContext[TADependency]) -> str:
        return repo_context(ctx, context_type="code")

    return agent


def create_unit_testing_agent() -> Agent[TADependency, UnitTestingResponse]:
    """Create an agent focused on unit testing evaluation."""
    agent = Agent(
        model=model,
        deps_type=TADependency,
        output_type=UnitTestingResponse,
        system_prompt="""You are a teaching assistant evaluating unit testing for a university MLOps course.

Evaluate testing based on:
- Test coverage (unit, integration, E2E tests)
- Test quality and assertions
- Use of testing frameworks (pytest, unittest, etc.)
- Mock usage and test isolation
- Test organization and naming conventions
- Edge case coverage

SCORE (1-5):
1: Poor - No or minimal tests, negligible coverage
2: Below Average - Tests exist but inadequate coverage
3: Average - Adequate coverage with noticeable gaps
4: Good - Good coverage of critical functionality
5: Excellent - Comprehensive test coverage

SUMMARY: 1-2 paragraphs (max 200 words) on testing findings and suggestions.
CONFIDENCE (1-10): Your confidence in the assessment.
""",
    )

    @agent.system_prompt
    async def add_repo_context(ctx: RunContext[TADependency]) -> str:
        return repo_context(ctx, context_type="tests")

    return agent


def create_cicd_agent() -> Agent[TADependency, CICDResponse]:
    """Create an agent focused on CI/CD evaluation."""
    agent = Agent(
        model=model,
        deps_type=TADependency,
        output_type=CICDResponse,
        system_prompt="""You are a teaching assistant evaluating CI/CD for a university MLOps course.

Evaluate CI/CD based on:
- GitHub Actions/workflow configuration
- Automated testing in pipelines
- Build and deployment automation
- Proper use of secrets and environment variables
- Pipeline efficiency and reliability
- Deployment strategies

SCORE (1-5):
1: Poor - No pipeline or completely broken
2: Below Average - Exists but unreliable or manual
3: Average - Functional but missing best practices
4: Good - Reliable and mostly automated
5: Excellent - Robust, fully automated, follows best practices

SUMMARY: 1-2 paragraphs (max 200 words) on CI/CD findings and suggestions.
CONFIDENCE (1-10): Your confidence in the assessment.
""",
    )

    @agent.system_prompt
    async def add_repo_context(ctx: RunContext[TADependency]) -> str:
        return repo_context(ctx, context_type="CI/CD configuration")

    return agent


def codebase(repo_link: str) -> TACodeResponse:
    code_quality_agent = create_code_quality_agent()
    unit_testing_agent = create_unit_testing_agent()
    cicd_agent = create_cicd_agent()

    try:
        # Code quality evaluation - focus on source code
        logger.info(f"Evaluating code quality for repository {repo_link}")
        code_quality_deps = TADependency(
            repo_link=repo_link,
            repomix=RepoMix(
                include=["**/*.py"],
                ignore=RepoMix.Ignore(
                    customPatterns=[
                        "tests/**",
                        "test_*.py",
                        "*_test.py",
                        ".github/**",
                        "**/*.ipynb",
                        "**/__pycache__/**",
                        "*.pyc",
                        "data/**",
                        "**/*.csv",
                        "reports/**",
                    ]
                ),
            ),
        )
        code_quality_result = code_quality_agent.run_sync(
            "Evaluate the code quality of this repository.",
            deps=code_quality_deps,
        )
        logger.info(f"Code quality score: {code_quality_result.output.score}")

        # Unit testing evaluation - focus on test files
        logger.info(f"Evaluating unit testing for repository {repo_link}")
        unit_testing_deps = TADependency(
            repo_link=repo_link,
            repomix=RepoMix(
                include=["tests/**/*.py", "test_*.py", "*_test.py", "**/*test*.py"],
                ignore=RepoMix.Ignore(
                    customPatterns=[
                        ".github/**",
                        "**/*.ipynb",
                        "**/__pycache__/**",
                        "*.pyc",
                    ]
                ),
            ),
        )
        unit_testing_result = unit_testing_agent.run_sync(
            "Evaluate the unit testing in this repository.",
            deps=unit_testing_deps,
        )
        logger.info(f"Unit testing score: {unit_testing_result.output.score}")

        # CI/CD evaluation - focus on workflow files
        logger.info(f"Evaluating CI/CD for repository {repo_link}")
        cicd_deps = TADependency(
            repo_link=repo_link,
            repomix=RepoMix(
                include=[
                    ".github/workflows/*.yml",
                    ".github/workflows/*.yaml",
                    "Dockerfile",
                    "**/Dockerfile*",
                    "docker-compose*.yml",
                    "docker-compose*.yaml",
                    ".dockerignore",
                ],
                ignore=RepoMix.Ignore(customPatterns=[]),
            ),
        )
        cicd_result = cicd_agent.run_sync(
            "Evaluate the CI/CD setup in this repository.",
            deps=cicd_deps,
        )
        logger.info(f"CI/CD score: {cicd_result.output.score}")

        # Aggregate results
        final_response = TACodeResponse.from_sub_agents(
            code_quality_response=code_quality_result.output,
            unit_testing_response=unit_testing_result.output,
            cicd_response=cicd_result.output,
        )

        logger.info(f"Overall score: {final_response.overall_score}")
        pprint(final_response)
    except Exception as e:
        logger.error(f"Failed for repository {repo_link}: {e}")
        shutil.rmtree(Path("output"))
        raise e
    shutil.rmtree(Path("output"))
    return final_response


def report(repo_link: str) -> TAReportResponse:
    """Main function to evaluate the report of a repository."""
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
    async def add_report_information(ctx: RunContext[TADependency]) -> str:
        repo_content = get_repo_content(ctx.deps.repo_link, ctx.deps.repomix)
        return f"""
        Report Content:\n\n
        {repo_content}
        """

    deps = TADependency(
        repo_link=repo_link, repomix=RepoMix(include=["reports/README.md"])
    )
    try:
        result = ta_agent.run_sync("What do you think of the groups report?", deps=deps)
        result.output.request_usage = result.usage()
        pprint(result.output)
    except Exception as e:
        shutil.rmtree(Path("output"))
        raise e
    shutil.rmtree(Path("output"))
    return result.output


if __name__ == "__main__":
    import typer

    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

    app.command()(codebase)
    app.command()(report)

    app()  # type: ignore
