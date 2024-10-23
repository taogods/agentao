"""
This is currently duplicated across taoception and gh-issue-pull repos
TODO: remove duplication
"""
from pydantic import BaseModel


class LabelledIssueTask(BaseModel):
    problem_statement: str
    patch: str
    s3_repo_url: str
