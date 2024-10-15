"""
This is currently duplicated across taoception and gh-issue-pull repos
TODO: remove duplication
"""

from pydantic import BaseModel
from typing import Optional

class SWEBenchEntry(BaseModel):
    instance_id: str
    text: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: Optional[str]
    created_at: str
    patch: str
