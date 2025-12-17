import csv
from pathlib import Path

from mlops_mentor.common.models import GroupInfo, RepoInfo


def load_groups(file_name: str = "group_info.csv") -> list[GroupInfo]:
    """Loads the group-repository data into a DataFrame."""
    with Path(file_name).open() as f:
        csv_reader = csv.reader(f, delimiter=",")
        content = []
        for i, row in enumerate(csv_reader):
            if i == 0:  # Skip the header
                continue
            group = GroupInfo(
                group_number=int(row[0]),
                student_1=row[1] if row[1] != "" else None,
                student_2=row[2] if row[2] != "" else None,
                student_3=row[3] if row[3] != "" else None,
                student_4=row[4] if row[4] != "" else None,
                student_5=row[5] if row[5] != "" else None,
                repo_info=RepoInfo(repo_url=row[6]),
            )
            content.append(group)
    return content
