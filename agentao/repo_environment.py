from typing import Dict, Final, List

from pydantic import BaseModel
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS


class RepoEnvironmentInfo(BaseModel):
    install_command: str
    python_version: str

    @property
    def config_dict(self) -> Dict:
        return dict(
            install=self.install_command,
            python=self.python_version,
        )

    @staticmethod
    def from_swebench(repo: str):
        if repo not in SUPPORTED_REPOS:
            raise ValueError(f"Repo {repo} is not supported. Must be one of: {SUPPORTED_REPOS}")

        specs_dict = MAP_REPO_VERSION_TO_SPECS[repo]

        # Get latest supported version
        max_key: str = max(specs_dict.keys(), key=lambda version: float(version))

        install_command: str = specs_dict[max_key]["install"]
        python_version: str = specs_dict[max_key]["python"]

        return RepoEnvironmentInfo(
            install_command=install_command,
            python_version=python_version
        )


SUPPORTED_REPOS: Final[List[str]] = [
    "mwaskom/seaborn",
    "pytest-dev/pytest",
    "pallets/flask",
    "pylint-dev/pylint",
    "psf/requests",
]

REPO_TO_ENVIRONMENT_INFO: Final[Dict[str, RepoEnvironmentInfo]] = {
    repo: RepoEnvironmentInfo.from_swebench(repo) for repo in SUPPORTED_REPOS
}
