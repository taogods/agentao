"""
This is currently duplicated across taoception and gh-issue-pull repos
TODO: remove duplication
"""
from pydantic import BaseModel


class LabelledIssueTask(BaseModel):
    """
    Task created from a closed (issue, PR) pair for normal
    miner incentive flow.
    """
    # Description of problem to be solved
    problem_statement: str
    # The solution patch
    patch: str
    # Link to s3 bucket containing the code
    s3_repo_url: str


class OpenIssueTask(BaseModel):
    """
    Task created from an open issue to be solved by miners.
    """
    # Link to issue containing problem statement
    issue_url: str
    # The url of the repository
    repo_url: str
    # The base commit that we want to merge from and into
    base_commit: str
    # Link to s3 bucket containing the code
    s3_repo_url: str
    # The name of the repository
    repo_name: str
    # The statement of the problem 
    problem_statement: str

class PendingRewards(BaseModel):
    """
    Miners who opened a PR and need to be rewarded.
    """
    # The amount of rewards pending
    uid: int
    # The url of the PR
    pr_url: str
    # TODO: Add state of acceptance: merge, draft accepted, etc
