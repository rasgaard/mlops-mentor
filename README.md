# MLOps Mentor 🤖

An automated teaching assistant for evaluating MLOps course projects using LLM-powered code analysis and GitHub repository scraping.

## Overview

MLOps Mentor helps teaching assistants evaluate student projects in Machine Learning Operations (MLOps) courses by:

- **Scraping** GitHub repositories for comprehensive metrics (commits, PRs, code structure, CI/CD status)
- **Analyzing** code quality, unit testing, and CI/CD practices using LLM judges
- **Visualizing** results through an interactive leaderboard dashboard

The tool automates the tedious parts of grading while providing detailed, consistent feedback on student submissions.


## Installation

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- GitHub Personal Access Token
- AI model access (Ollama or CampusAI)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rasgaard/mlops-mentor.git
cd mlops-mentor
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Configure environment variables:
```bash
cp .env.template .env
```

4. Prepare your `group_info.csv` file with student repository URLs:
```csv
group_nb,student 1,student 2,student 3,student 4,student 5,github_repo
1, s123456, s654321, , , ,https://github.com/user/repo1
2, s111111, s222222, s333333, , ,https://github.com/user/repo2
```

## Usage

### Running the Full Pipeline

Evaluate all repositories in `group_info.csv`:
```bash
uv run --env-file .env ./src/mlops_mentor/run.py
```


## Configuration

### Evaluation Criteria

Each LLM agent evaluates specific aspects:

1. **Code Quality** (1-5 scale):
   - Code structure and organization
   - Python best practices (PEP 8, type hints, docstrings)
   - Readability and maintainability
   - Design patterns and configuration management

2. **Unit Testing** (1-5 scale):
   - Test coverage (unit, integration, E2E)
   - Test quality and assertions
   - Framework usage (pytest, unittest)
   - Mock usage and test isolation

3. **CI/CD Practices** (1-5 scale):
   - Automation setup (GitHub Actions, etc.)
   - Pipeline quality and best practices
   - Testing and deployment automation
   - Documentation and configuration


## Authors

* Nicki Skafte Detlefsen (nsde@dtu.dk)
* Rasmus Aagaard (roraa@dtu.dk)
