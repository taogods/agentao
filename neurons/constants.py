from typing import Final

from neurons.classes import OpenIssueTask, LabelledIssueTask

NO_MINER_RESPONSE_SCORE: float = 0.005
DATA_ENDPOINT_BY_TASK = {
    OpenIssueTask: "https://gh-issue-pull.onrender.com/open_issue",
    LabelledIssueTask: "https://gh-issue-pull.onrender.com/synthetic_task"
}
PENDING_REWARDS_ENDPOINT: Final[str] = "https://gh-issue-pull.onrender.com/pending_rewards"
REGISTER_PR_ENDPOINT: Final[str] = "https://gh-issue-pull.onrender.com/register_pr"
UPLOAD_ISSUE_ENDPOINT: Final[str] = "https://gh-issue-pull.onrender.com/upload_issue"
DOCKER_CACHE_LEVEL: Final[str] = "instance"
