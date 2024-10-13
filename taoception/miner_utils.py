from dataclasses import dataclass
from getpass import getuser
from pathlib import Path

from sweagent.agent.agents import Agent
from sweagent.agent.models import ModelArguments
from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv

from SWEAgent.run import ActionsArguments, ScriptArguments, AgentArguments


@dataclass
class UnsolvedIssue:
    desc: str
    code_link: str

@dataclass
class IssueSolution:
    patch: str

def create_script_arguments(unsolved_issue: UnsolvedIssue) -> ScriptArguments:
    return ScriptArguments(
        environment=EnvironmentArguments(
            image_name="sweagent/swe-agent:latest",
            data_path=f"text://{unsolved_issue.desc}",
            repo_path="https://github.com/SWE-agent/test-repo",
            verbose=True,
        ),
        skip_existing=False,
        agent=AgentArguments(
            model=ModelArguments(
                # TODO(alex): Make model configurable
                model_name="claude-sonnet-3.5",
            ),
            config_file=Path("SWEAgent/config/default_from_url.yaml"),
        ),
        actions=ActionsArguments(
            open_pr=False,
            skip_if_commits_reference_issue=False,
            apply_patch_locally=True,
        ),
        print_config=True,
    )

def generate_code_patch(unsolved_issue: UnsolvedIssue) -> IssueSolution:
    script_arguments = create_script_arguments(unsolved_issue)

    env = SWEEnv(script_arguments.environment)
    observation, info = env.reset(0)

    agent = Agent("primary", script_arguments.agent)
    info, trajectory = agent.run(
        setup_args={"issue": getattr(env, "query", None), "files": [], "test_files": [], "tests": []},
        env=env,
        observation=observation,
        traj_dir=Path("trajectories") / Path(getuser()) / script_arguments.run_name,
        return_type="info_trajectory",
    )
    return IssueSolution(patch=info["submission"])
