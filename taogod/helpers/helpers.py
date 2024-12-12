import csv
import itertools
import json
import shutil
import textwrap
from pathlib import Path
from pprint import pformat
from statistics import mean
from typing import List, Dict, Union, TypedDict

import yaml
from git import Repo
from tabulate import tabulate

from taogod.helpers.classes import FilePair, FloatGraderScore, FullyScoredProblem, convert_to_obj
from taogod.helpers.clients import logger
from taogod.helpers.constants import PRICING_DATA_PER_MILLION_TOKENS, SENTINEL_FLOAT_FAILURE_VALUE, \
    SENTINEL_STRING_FAILURE_VALUE


def clone_repo(author_name: str, repo_name: str, base_path: Path) -> Path:
    """
    Clone a GitHub repository to a specified directory under 'repos' and return the path.

    :param author_name: GitHub username or organization name.
    :param repo_name: Repository name.
    :param base_path: Base path where the 'repos' directory will be created.
    :return: Path to the cloned repository.
    """
    try:
        repos_dir = base_path / "repos"
        repos_dir.mkdir(parents=True, exist_ok=True)

        clone_to_path = repos_dir / repo_name
        if clone_to_path.exists() and clone_to_path.is_dir():
            shutil.rmtree(clone_to_path)
            logger.info(f"Directory {clone_to_path} has been removed.")

        Repo.clone_from(f"https://github.com/{author_name}/{repo_name}.git", clone_to_path)
        logger.info(f"Repository cloned to {clone_to_path}")
        return clone_to_path
    except Exception:
        logger.exception(f"Failed to clone repository")
        raise



def compute_overall_score(miner_output_score: FloatGraderScore) -> float:
    DYNAMIC_CHECKLIST_WEIGHT = 0.2
    ADDRESSES_PROBLEM_WEIGHT = 0.3
    LOGICAL_SOLUTION_WEIGHT = 0.25
    BREVITY_WEIGHT = 0.05
    POTENTIAL_BUGS_WEIGHT = 0.2

    static_score = (
        ADDRESSES_PROBLEM_WEIGHT * miner_output_score.addresses_problem_in_statement +
        LOGICAL_SOLUTION_WEIGHT * miner_output_score.logical_solution +
        BREVITY_WEIGHT * miner_output_score.brevity_and_cleanliness_of_code +
        POTENTIAL_BUGS_WEIGHT * (1 - miner_output_score.potential_bugs_generated)
    )

    if not miner_output_score.dynamic_checklist_scores:
        return static_score / (1. - DYNAMIC_CHECKLIST_WEIGHT)

    return (
        static_score +
        DYNAMIC_CHECKLIST_WEIGHT * mean(miner_output_score.dynamic_checklist_scores)
    )


def calculate_price(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing_dict = PRICING_DATA_PER_MILLION_TOKENS[model_name]
    input_price, output_price = pricing_dict["input"], pricing_dict["output"]
    return (input_tokens * input_price + output_tokens * output_price) / 1e6


def highest_cosine_filepair_selector(file_pairs: List[FilePair]) -> FilePair:
    selected_file_pair = sorted(
        file_pairs,
        key=lambda x: float(x.cosine_similarity),
        reverse=True
    )[0]

    return selected_file_pair


def parse_yaml(config_path: Path) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Found config {pformat(config)}")
    return config


def save_full_data(solutions: Dict[str, List[FullyScoredProblem]], file_path: Path = Path("full_eval_data.json")) -> None:
    full_data: List[Dict[str, List[TypedDict]]] = []

    if file_path.exists() and file_path.is_file():
        with open(file_path, 'r') as file:
            try:
                full_data = json.load(file)
                if not isinstance(full_data, list):
                    raise ValueError("Existing data is not a list. Cannot append.")
            except json.JSONDecodeError:
                pass

    full_data.append(convert_to_obj(solutions))

    # Write back to file
    with open(file_path, 'w') as file:
        json.dump(full_data, file, indent=4)


def save_display_data(data: List[List[Union[float, int, str]]], file_path: Path = Path("solutions.csv")) -> None:
    """
    Save or append the given data to a CSV file.

    :param data: A list of rows where each row is a list of values.
    :param file_path: The path to the CSV file.
    """
    headers = [
        "Repo",
        "Problem",
        "Model",
        "Solution Patch",
        "Output Score",
        "Miner $",
        "Validator $",
        "Duration (s)",
        "Miner $/min",
    ]

    # Check if file exists
    file_exists = file_path.is_file()

    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers if the file does not exist
        if not file_exists:
            writer.writerow(headers)

        # Write the data rows
        writer.writerows(data)

flatten = lambda x: list(itertools.chain.from_iterable(x))

def repeat_list(lst: List, num_repeats: int) -> List:
    return flatten([lst[:] for _ in range(num_repeats)])


def flatten_and_display_solutions(solutions: Dict[str, List[FullyScoredProblem]], should_save_data: bool=True) -> None:
    # Helper to wrap text for better display
    def wrap_text(text, width=50):
        return "\n".join(textwrap.wrap(text, width=width))

    # Flatten the solutions dictionary
    flat_data = []
    for repo, problems in solutions.items():
        for problem in problems:
            problem: FullyScoredProblem
            overall_score = SENTINEL_FLOAT_FAILURE_VALUE
            if problem.miner_output_score is not None:
                overall_score = compute_overall_score(problem.miner_output_score)

            time_to_solve_s = problem.time_to_solve_s

            validator_cost = SENTINEL_FLOAT_FAILURE_VALUE
            if problem.generated_problem_statement.model_stats is not None:
                validator_cost = problem.generated_problem_statement.model_stats.cost

            miner_cost, miner_cost_per_min = SENTINEL_FLOAT_FAILURE_VALUE, SENTINEL_FLOAT_FAILURE_VALUE
            miner_solution_patch = SENTINEL_STRING_FAILURE_VALUE
            miner_exit_status = SENTINEL_STRING_FAILURE_VALUE
            if problem.miner_solution is not None and problem.miner_solution.model_stats is not None:
                miner_cost = problem.miner_solution.model_stats.total_cost
                miner_cost_per_min = miner_cost / time_to_solve_s * 60.
                miner_solution_patch = problem.miner_solution.patch
                miner_exit_status = problem.miner_solution.exit_status

            flat_data.append([
                wrap_text(repo, width=30),
                wrap_text(problem.generated_problem_statement.problem_statement[:100] + "...", width=50),
                problem.miner_llm,
                wrap_text(miner_solution_patch[:100] + "...", width=50),
                wrap_text(str(overall_score), width=50),
                f"{miner_cost:.2f}",
                f"{validator_cost:.2f}",
                f"{time_to_solve_s:.2f}",
                f"{miner_cost_per_min:.2f}",
                f"{miner_exit_status}"
            ])

    # Define headers
    headers = [
        "Repo",
        "Problem",
        "Model",
        "Solution Patch",
        "Output Score",
        "Miner $",
        "Validator $",
        "Duration (s)",
        "Miner $/min",
        "Exit Status"
    ]

    if should_save_data:
        save_display_data(flat_data)
        save_full_data(solutions)

    logger.info(tabulate(flat_data, headers=headers, tablefmt="fancy_grid", stralign="left"))
