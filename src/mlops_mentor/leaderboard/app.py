import base64
from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset


def create_matplotlib_sparkline(contributions):
    """Create a tiny inline sparkline chart using matplotlib."""
    if not contributions or not isinstance(contributions, list):
        return ""

    # Create figure with minimal size
    fig, ax = plt.subplots(figsize=(1.5, 0.4))

    # Plot bars
    ax.bar(range(len(contributions)), contributions, color="#37536d", width=0.8)

    # Remove all axes and borders
    ax.axis("off")
    ax.margins(0)

    # Tight layout to minimize whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Convert to base64 encoded image
    buffer = BytesIO()
    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=50,
        transparent=True,
    )
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()

    return f'<img src="data:image/png;base64,{img_base64}" style="vertical-align: middle; height: 20px;"/>'


def create_text_sparkline(contributions):
    """Create a text-based sparkline using Unicode block characters."""
    if not contributions or not isinstance(contributions, list):
        return ""

    if not contributions:
        return ""

    max_val = max(contributions) if contributions else 1
    blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    sparkline = "".join(
        [
            blocks[min(int((val / max_val) * (len(blocks) - 1)), len(blocks) - 1)]
            for val in contributions
        ]
    )

    return sparkline


def format_boolean(value):
    """Convert boolean to emoji."""
    if pd.isna(value):
        return "❓"
    return "✅" if value else "❌"


def load_leaderboard_data():
    """Load the repo stats dataset from Hugging Face."""
    dataset = load_dataset("rasgaard/mlops-mentor-stats", "repo_stats")
    df = pd.DataFrame(dataset["train"])

    # Sort by relevant metrics
    df = df.sort_values("total_commits", ascending=False)

    # Round numeric columns
    df["repo_size"] = df["repo_size"].round(2)
    df["average_commit_length_to_main"] = df["average_commit_length_to_main"].round(1)

    # Add sparkline column
    df["contrib_distribution"] = df["contributions_per_contributor"].apply(
        create_matplotlib_sparkline
    )

    # Convert boolean columns to emoji
    boolean_columns = [
        "has_requirements_file",
        "has_cloudbuild",
        "using_dvc",
        "actions_passing",
    ]
    for col in boolean_columns:
        df[col] = df[col].apply(format_boolean)

    # Select and reorder columns for better display
    display_columns = [
        "group_number",
        "group_size",
        "num_contributors",
        "contrib_distribution",
        "total_commits",
        "num_prs",
        "num_commits_to_main",
        "average_commit_length_to_main",
        "latest_commit",
        "num_docker_files",
        "num_python_files",
        "num_workflow_files",
        "has_requirements_file",
        "has_cloudbuild",
        "using_dvc",
        "actions_passing",
        "num_warnings",
        "repo_size",
        "readme_length",
    ]

    return df[display_columns]


def create_leaderboard():
    """Create the Gradio leaderboard interface."""
    with gr.Blocks(title="MLOps Mentor Leaderboard") as demo:
        gr.Markdown("# 🏆 MLOps Mentor Leaderboard")
        gr.Markdown("Repository statistics for MLOps course groups")

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Data", variant="primary")

        leaderboard_table = gr.Dataframe(
            value=load_leaderboard_data(),
            interactive=False,
            wrap=True,
            label="Repository Statistics",
            datatype=[
                "number",
                "number",
                "number",
                "html",
                "number",
                "number",
                "number",
                "number",
                "str",
                "number",
                "number",
                "number",
                "str",  # has_requirements_file (emoji)
                "str",  # has_cloudbuild (emoji)
                "str",  # using_dvc (emoji)
                "str",  # actions_passing (emoji)
                "number",
                "number",
                "number",
            ],
        )

        refresh_btn.click(fn=load_leaderboard_data, outputs=leaderboard_table)

        gr.Markdown(
            """
        ### Metrics Explained
        - **group_number**: Group identifier
        - **contrib_distribution**: Sparkline showing contribution distribution per contributor
        - **total_commits**: Total number of commits across all branches
        - **num_commits_to_main**: Commits to the main branch
        - **num_prs**: Number of pull requests
        - **actions_passing**: ✅ if GitHub Actions are passing, ❌ if failing
        - **using_dvc**: ✅ if repository uses DVC, ❌ if not
        - **has_requirements_file**: ✅ if requirements file exists, ❌ if not
        - **has_cloudbuild**: ✅ if cloudbuild config exists, ❌ if not
        - **num_warnings**: Number of warnings detected
        """
        )

    return demo


if __name__ == "__main__":
    demo = create_leaderboard()
    demo.launch(share=False)
