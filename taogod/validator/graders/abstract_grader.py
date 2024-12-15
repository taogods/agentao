from abc import ABC
from typing import List

from pydantic import BaseModel

from taogod.helpers.classes import GeneratedProblemStatement, IssueSolution


class MinerSubmission(BaseModel):
    repo: str
    problem: GeneratedProblemStatement
    solution: IssueSolution


class GraderInterface(ABC):
    def grade(self, submissions: List[MinerSubmission]) -> List[float]:
        raise NotImplementedError("AbstractGrader.grade() must be overridden")

