"""
This is currently duplicated across taoception and gh-issue-pull repos
TODO: remove duplication
"""
from pydantic import BaseModel
from typing import Optional, List


class SWEBenchEntry(BaseModel):
    instance_id: str
    text: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: Optional[str]
    created_at: str
    patch: str
    test_patch: str
    FAIL_TO_PASS: List[str]
    PASS_TO_PASS: List[str]

