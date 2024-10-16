"""
This is currently duplicated across taoception and gh-issue-pull repos
TODO: remove duplication
"""
from pydantic import BaseModel
from typing import Optional, List
from swebench.harness.constants import SWEbenchInstance


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

    def to_swebench_instance(self) -> SWEbenchInstance:
        return SWEbenchInstance(
            instance_id=self.instance_id,
            text=self.text,
            repo=self.repo,
            base_commit=self.base_commit,
            problem_statement=self.problem_statement,
            hints_text=self.hints_text,
            created_at=self.created_at,
            patch=self.patch,
            test_patch=self.test_patch,
            FAIL_TO_PASS=self.FAIL_TO_PASS,
            PASS_TO_PASS=self.PASS_TO_PASS,
            environment_setup_commit="",
        )

