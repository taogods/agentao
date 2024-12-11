import logging
from dataclasses import dataclass
from pathlib import Path

from sweagent.agent.agents import Agent, AgentArguments
from sweagent.agent.models import ModelArguments
from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv

from .swe_agent_arguments import ScriptArguments, ActionsArguments

logger = logging.getLogger(__name__)

@dataclass
class UnsolvedIssue:
    desc: str
    local_code_path: Path

@dataclass
class IssueSolution:
    patch: str

def create_script_arguments(
        model_name: str,
        unsolved_issue: UnsolvedIssue
    ) -> ScriptArguments:
    swe_agent_root = Path("SWE-agent")
    return ScriptArguments(
        environment=EnvironmentArguments(
            image_name="sweagent/swe-agent:latest",
            data_path=f"text://{unsolved_issue.desc}",
            repo_path=str(unsolved_issue.local_code_path),
            environment_setup=swe_agent_root / "config/environment_setup/seaborn.yaml",
            verbose=True,
            install_environment=True,
        ),
        skip_existing=False,
        agent=AgentArguments(
            model=ModelArguments(
                model_name= model_name,
                per_instance_cost_limit=instance_cost,
            ),
            config_file=Path(swe_agent_root / "config/default_from_url.yaml"),
        ),
        actions=ActionsArguments(
            open_pr=False,
            skip_if_commits_reference_issue=False,
            apply_patch_locally=True,
        ),
        print_config=True,
    )

def generate_code_patch(
        model_name: str,
        instance_cost: float,
    ) -> IssueSolution:
    script_arguments = create_script_arguments(model_name, instance_cost)

    env = SWEEnv(script_arguments.environment)
    observation, info = env.reset(0)

    agent = Agent("primary", script_arguments.agent)
    trajectories_dir = Path.cwd() / "trajectories"
    trajectories_dir.mkdir(exist_ok=True)

    logger.info("Running sweagent...")
    info, trajectory = agent.run(
        setup_args={"issue": getattr(env, "query", None), "files": [], "test_files": [], "tests": []},
        env=env,
        observation=observation,
        traj_dir=trajectories_dir,
        return_type="info_trajectory",
    )
    logger.info(f"Finished running sweagent. Received info: {info}")
    return IssueSolution(patch=info["submission"])