import json
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai.usage import RunUsage

from mlops_mentor.common.models import GroupInfo


class RepoMix(BaseModel):
    """Configuration for the repomix."""

    class Output(BaseModel):
        """Output configuration for the repomix."""

        filePath: str = "repomix-output.md"  # noqa: N815
        style: str = "markdown"
        removeComments: bool = False  # noqa: N815
        removeEmptyLines: bool = False  # noqa: N815
        showLineNumbers: bool = False  # noqa: N815
        copyToClipboard: bool = False  # noqa: N815
        topFilesLength: int = 10  # noqa: N815

    class Ignore(BaseModel):
        """Ignore configuration for the repomix."""

        useGitignore: bool = True  # noqa: N815
        useDefaultPatterns: bool = True  # noqa: N815
        customPatterns: list = []  # noqa: N815

    class Security(BaseModel):
        """Security configuration for the repomix."""

        enableSecurityCheck: bool = True  # noqa: N815

    output: Output = Output()
    include: list = ["**/*"]
    ignore: Ignore = Ignore()
    security: Security = Security()

    def dump_json(self, file_path: str) -> None:
        """Dump the configuration to a JSON file."""
        with Path(file_path).open("w") as file:
            json.dump(self.model_dump(), file, indent=4)


class TADependency(BaseModel):
    """Model for the dependencies of the TA agent."""

    group_info: GroupInfo
    repomix: RepoMix


class CodeQualityResponse(BaseModel):
    """Model for the response from the code quality agent."""

    score: int = Field(
        ..., ge=1, le=5, description="Score the code quality on a scale from 1 to 5"
    )
    summary: str = Field(
        ..., description="Brief summary of code quality findings (max 200 words)"
    )
    confidence: int = Field(
        ..., ge=1, le=10, description="Confidence in the score from 1-10"
    )


class UnitTestingResponse(BaseModel):
    """Model for the response from the unit testing agent."""

    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score the unit testing in the codebase on a scale from 1 to 5",
    )
    summary: str = Field(
        ..., description="Brief summary of unit testing findings (max 200 words)"
    )
    confidence: int = Field(
        ..., ge=1, le=10, description="Confidence in the score from 1-10"
    )


class CICDResponse(BaseModel):
    """Model for the response from the CI/CD agent."""

    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score the continuous integration and deployment process on a scale from 1 to 5",
    )
    summary: str = Field(
        ..., description="Brief summary of CI/CD findings (max 200 words)"
    )
    confidence: int = Field(
        ..., ge=1, le=10, description="Confidence in the score from 1-10"
    )


class TACodeResponse(BaseModel):
    """Model for the response from the TA agent for the code."""

    code_quality: int = Field(
        ..., ge=1, le=5, description="Score the code quality on a scale from 1 to 5"
    )
    unit_testing: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score the unit testing in the codebase on a scale from 1 to 5",
    )
    ci_cd: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score the continuous integration and deployment process on a scale from 1 to 5",
    )
    summary: str = Field(
        ..., description="Provide a brief summary of the code quality and unit testing"
    )
    overall_score: int = Field(
        ..., ge=1, le=10, description="Overall score from 1-10 for the whole codebase"
    )
    confidence: int = Field(
        ..., ge=1, le=10, description="Confidence in the overall score from 1-10"
    )

    @classmethod
    def from_sub_agents(
        cls,
        code_quality_response: CodeQualityResponse,
        unit_testing_response: UnitTestingResponse,
        cicd_response: CICDResponse,
    ) -> "TACodeResponse":
        """Create TACodeResponse from individual sub-agent responses."""
        # Calculate overall score as weighted average
        overall_score = round(
            (
                code_quality_response.score * 2
                + unit_testing_response.score * 2
                + cicd_response.score * 1.5
            )
            / 1.1
        )
        overall_score = max(1, min(10, overall_score))  # Ensure 1-10 range

        # Calculate confidence as average of sub-agent confidences
        avg_confidence = round(
            (
                code_quality_response.confidence
                + unit_testing_response.confidence
                + cicd_response.confidence
            )
            / 3
        )

        # Combine summaries
        summary = (
            f"Code Quality: {code_quality_response.summary}\n\n"
            f"Unit Testing: {unit_testing_response.summary}\n\n"
            f"CI/CD: {cicd_response.summary}"
        )

        return cls(
            code_quality=code_quality_response.score,
            unit_testing=unit_testing_response.score,
            ci_cd=cicd_response.score,
            summary=summary,
            overall_score=overall_score,
            confidence=avg_confidence,
        )


class TAReportResponse(BaseModel):
    """Model for the response from the TA agent for the report."""

    checklist: int = Field(
        ..., ge=0, le=52, description="How many items from the checklist were completed"
    )
    coding_env: int = Field(
        ..., ge=1, le=5, description="Score for section on coding environment"
    )
    version_control: int = Field(
        ..., ge=1, le=5, description="Score for section on version control"
    )
    code_run_and_experiments: int = Field(
        ..., ge=1, le=5, description="Score for section on code run and experiments"
    )
    cloud: int = Field(..., ge=1, le=5, description="Score for section on cloud")
    deployment: int = Field(
        ..., ge=1, le=5, description="Score for section on deployment"
    )
    overall_discussion: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score for section on overall discussion of project",
    )
    summary: str = Field(
        ..., description="Provide a brief summary of the report evaluation"
    )
    overall_score: int = Field(
        ..., ge=1, le=10, description="Overall score from 1-10 for report"
    )
    confidence: int = Field(
        ..., ge=1, le=10, description="Confidence in the overall score from 1-10"
    )

    request_usage: RunUsage | None = None
