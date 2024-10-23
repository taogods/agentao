from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from sweagent.agent.agents import AgentArguments
from sweagent.environment.swe_env import EnvironmentArguments
from sweagent.environment.utils import (
    get_data_path_name,
)


@dataclass(frozen=True)
class ActionsArguments(FlattenedAccess, FrozenSerializable):
    """Run real-life actions (opening PRs, etc.) if we can solve the issue."""

    # Open a PR with the patch if we can solve the issue
    open_pr: bool = False
    # When working with local repository: Apply patch
    apply_patch_locally: bool = False
    # Option to be used with open_pr: Skip action if there are already commits claiming
    # to fix the issue. Please only set this to False if you are sure the commits are
    # not fixes or if this is your own repository!
    skip_if_commits_reference_issue: bool = True
    # OBSOLETE. Do not use, will raise error. Please specify --repo_path instead.
    push_gh_repo_url: str = ""

    def __post_init__(self):
        if self.push_gh_repo_url:
            msg = "push_gh_repo_url is obsolete. Use repo_path instead"
            raise ValueError(msg)


@dataclass(frozen=True)
class ScriptArguments(FlattenedAccess, FrozenSerializable):
    """Configure the control flow of the run.py script"""

    environment: EnvironmentArguments
    agent: AgentArguments
    actions: ActionsArguments
    # Only run instances that completely match this regex
    instance_filter: str = ".*"
    # Skip instances with existing trajectories
    skip_existing: bool = True
    # Suffix for the run name (used for example in trajectory directory naming)
    suffix: str = ""
    # Raise unhandled exceptions during the run (useful for debugging)
    raise_exceptions: bool = False
    # Dump the entire config to the log
    print_config: bool = True
    # Run the agent in CTF mode (SWE-agent: EnIGMA)
    ctf: bool = False

    @property
    def run_name(self) -> str:
        """Generate a unique name for this run based on the arguments."""
        model_name = self.agent.model.model_name.replace(":", "-")
        data_stem = get_data_path_name(self.environment.data_path)
        assert self.agent.config_file is not None  # mypy
        config_stem = Path(self.agent.config_file).stem

        temp = self.agent.model.temperature
        top_p = self.agent.model.top_p

        per_instance_cost_limit = self.agent.model.per_instance_cost_limit
        install_env = self.environment.install_environment

        return (
            f"{model_name}__{data_stem}__{config_stem}__t-{temp:.2f}__p-{top_p:.2f}"
            + f"__c-{per_instance_cost_limit:.2f}__install-{int(install_env)}"
            + (f"__{self.suffix}" if self.suffix else "")
        )