# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Taogod

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
from datetime import timedelta
from pathlib import Path
from textwrap import dedent
from typing import *

import math
import numpy as np
from aiohttp import BasicAuth, ClientSession
from jinja2 import Template
from sweagent.environment.swe_env import SWEEnv

from neurons.constants import UPLOAD_ISSUE_ENDPOINT
from neurons.helpers import logger
from taogod.base.validator import BaseValidatorNeuron, TaskType
from taogod.helpers.classes import GeneratedProblemStatement, ProblemGeneratorParameters, ValidatorModelStats, \
    IngestionHeuristics, IssueSolution
from taogod.helpers.constants import SENTINEL_FLOAT_FAILURE_VALUE, SENTINEL_INT_FAILURE_VALUE, \
    SENTINEL_STRING_FAILURE_VALUE
from taogod.helpers.helpers import clone_repo, highest_cosine_filepair_selector, compute_overall_score
from taogod.protocol import CodingTask
from taogod.synthetic_testing import apply_patch, compare_test_results, run_tests
from taogod.utils.uids import check_uid_availability
from taogod.validator.generate_problem import generate_problem_statements
from taogod.validator.grade_output import grade_miner_solution
from taogod.validator.ingest import get_all_filepairs

CODINGTASK_TIMEOUT_MINS: Final[float] = 30.

PROBLEM_STATEMENT_TEMPLATE: Final[Template] = Template(
    dedent("""
    You are a skilled software engineering assistant. You will be provided with multiple files as context. Each file will contain portions of code, documentation, or relevant information about a software system. Your task is to come up with a specific software engineering problem that requires a solution to involve at least two of these files. You will generate a list of these problems, in the generated_problems array response.

    Further, once you have a problem statement, generate a checklist of points to consider and things that should be present in the solution (for example, are the correct Github API calls made if its a function that interfaces with the api). Generate several of these into dynamic_checklist field.
    Some additional guidelines are:
    - Do not output anything other than software engineering problem
    - The problem description should be very detailed and meticulous. It should contain sufficient context such that someone equipped with the codebase and your problem statement will have enough information to implement
    - The problem should be solvable by an autonomous SWE which can do things like installing PyPi packages, but cannot do things like make cloud provider accounts and register for other services manually.
    - The problem should not be overly difficult to implement, and should be fairly easy and not take too many LLM calls. 
    - Do not disclose which files would need to be modified to solve the problem.

    Here are the files:
    {% for file in files %}
    Filename: {{ file.path }}
    ```python3
    {{ file.contents }}
    ```
    {% endfor %}
    ```
    """)
)

GRADER_SYSTEM_PROMPT: Final[str] = """
Instructions:
    You are tasked with evaluating a code patch to determine how well it addresses a specific problem. Please follow these steps:
    Read the Problem Statement to understand the issue that needs to be resolved.
    Review the Git Diff to see the changes introduced by the patch.
    Examine the Affected Files to understand the context of the changes.
Your Task:
    Assess the patch for correctness, completeness, and effectiveness in solving the problem.
    Fill out each field (addresses problem in statement, whether its a logical or dumb solution, brevity and how clean the code is, and how likely it is to introduce other bugs)
    Consider any potential side effects or issues introduced by the patch.
    Grade a concise solution higher than a lengthy one assuming both are correct and complete.
    Provide a percentage numerical score between 0 and 1 representing how well the patch solves the problem:
    1 means the patch perfectly and completely solves the problem.
    0 means the patch does not address the problem at all.
    If you do not know for sure that the patch perfectly and completely solved the problem, do not give it 1. Instead, give it some value between 0 and 1. Be harshly critical of the submissions you receive, think carefully to find ways in which they may have issues, and make sure the score is reduced appropriately. You will be penalized more harshly if you give scores that are too high than scores that are too low, so bias on the side of giving lower scores.
"""


def exponential_decay(N, x):
    """
    Outputs a value that approaches 1 as x approaches 0 and approaches 0 as x approaches or exceeds N.

    Parameters:
    - N (int or float): The threshold value.
    - x (int or float): The input value.

    Returns:
    - float: The output value.
    """
    if x >= N:
        return 0
    return math.exp(-x / (N - x))

def create_problem_statements(
    validator_llm: str,
    repo: str,
    local_repo_dir: Path,
    problems: Union[int, List[str]],
    ingestion_heuristics: IngestionHeuristics
) -> List[GeneratedProblemStatement]:
    if isinstance(problems, int):
        problem_generator_params = ProblemGeneratorParameters(
            filepair_selection_logic=highest_cosine_filepair_selector,
            prompt_template=PROBLEM_STATEMENT_TEMPLATE,
            num_problems_to_gen=problems,
            problem_gen_model=validator_llm,
        )

        problem_statements: List[GeneratedProblemStatement] = generate_problems_for_single_repo(
            repo_path=local_repo_dir,
            ingestion_heuristics=ingestion_heuristics,
            problem_generation_params=problem_generator_params
        )

    elif isinstance(problems, list) and all(isinstance(text, str) for text in problems):
        problem_statements: List[GeneratedProblemStatement] = [
            GeneratedProblemStatement(
                prompt=SENTINEL_STRING_FAILURE_VALUE,
                model=SENTINEL_STRING_FAILURE_VALUE,
                problem_statement=text,
                dynamic_checklist=[],
                model_stats=ValidatorModelStats(
                    input_tokens=SENTINEL_INT_FAILURE_VALUE,
                    output_tokens=SENTINEL_INT_FAILURE_VALUE,
                    cost=SENTINEL_FLOAT_FAILURE_VALUE,
                )
            ) for text in problems
        ]


    else:
        raise ValueError(
            f"config[{repo}]['problems'] must be a list of strings or an integer. "
            f"Current value of `{problems}` is invalid"
        )
    return problem_statements


def generate_problems_for_single_repo(
    repo_path: Path,
    ingestion_heuristics: IngestionHeuristics,
    problem_generation_params: ProblemGeneratorParameters
) -> List[GeneratedProblemStatement]:
    file_pairs = get_all_filepairs(
        repo_path,
        heuristics=ingestion_heuristics,
        refresh=False
    )

    # Generate one problem statement, with prompt and model to benchmark
    problem_statements_list = generate_problem_statements(
        filepairs=file_pairs,
        parameters=problem_generation_params
    )
    return problem_statements_list


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        logger.info("load_state()")
        self.load_state()

        # TODO(developer): Anything specific to your use case you can do here

    @staticmethod
    def process_response_wrapper(args):
        """
        A wrapper function to handle multiprocessing safely.
        Takes a tuple of arguments to pass to the `process_response` function.
        """
        response, env_args, test_patch, tests_before = args
        try:
            env = SWEEnv(env_args)
            env.reset(0)
            # Apply patches
            apply_patch(env, response)

            # Run tests after applying patches
            tests_after = run_tests(env)

            # Compare test results
            results = compare_test_results(tests_before, tests_after)
            return results
        except Exception as e:
            logger.exception(f"Error in synthetic rewards: {e}")
            return None

    @staticmethod
    async def calculate_rewards(
        repo: str,
        problem: GeneratedProblemStatement,
        issue_solutions: List[IssueSolution],
        process_times: List[float],
    ) -> np.ndarray:
        """
        Validate the responses from the miners. This function should score the responses and return a list of rewards for each miner.
        """
        llm_evals = np.array([
            compute_overall_score(grade_miner_solution(
                repo,
                problem,
                issue_solution,
            ))
            for issue_solution in issue_solutions
        ])

        response_times = np.array([
            exponential_decay(CODINGTASK_TIMEOUT_MINS*60, time)
            for time in process_times
        ])

        # Commented out for now
        # ## Synthetic testing
        # env_args = EnvironmentArguments(
        #         image_name="sweagent/swe-agent:latest",
        #         data_path="text://example.json", # Doesnt matter for tests
        #         repo_path=str(codebase),
        #         verbose=True,
        #         environment_setup=str(env_setup_path),
        #     )
        # env = SWEEnv(env_args)
        # env.reset(0)
        #
        # tests_before = run_tests(env)
        #
        # # Share `tests_before` and other data across processes by making them part of the input arguments
        # tasks = [(response, env_args, tests_before) for response in responses]
        #
        # with ThreadPoolExecutor() as executor:
        #     synthetic_tests = list(executor.map(Validator.process_response_wrapper, tasks))
        #
        # syn_tests_arr = np.array([])
        # for test in synthetic_tests:
        #     if test is None: np.append(syn_tests_arr, 0.0)
        #     else:
        #         syn_tests_arr = np.append(
        #             syn_tests_arr,
        #             2 * int(len(test["PASS_TO_FAIL"]) == 0)
        #             + int(len(test["FAIL_TO_PASS"]) >= 0)
        #             + 3 * int(len(test["NEW_FAIL"]) == 0)
        #         )
        #
        # os.remove(env_setup_path)

        return llm_evals + response_times
    
    # TODO: Add more fields once components of scoring are named
    async def upload_solution(
            self,
            problem_statement: str,
            responses: List[IssueSolution],
            rewards_list: List[float],
            hotkeys: List[str],
    ):
        """
        Upload the closed issue to the data endpoint.
        """
        response_patches = [response.patch for response in responses]

        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"
        try:
            async with ClientSession() as session:
                # TODO: Add how long it takes to upload the issue
                payload = [{
                    "problem_statement": problem_statement,
                    "solution_patch": response_patch,
                    "score": response_score,
                    "miner_hotkey": miner_hotkey,
                } for
                    response_patch,
                    response_score,
                    miner_hotkey
                    in zip(response_patches, rewards_list, hotkeys)
                ]
                async with session.post(
                    url=UPLOAD_ISSUE_ENDPOINT,
                    auth=BasicAuth(hotkey, signature),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    _result = await response.json()
        except Exception:
            logger.exception("Error uploading closed issue")
        ...
    
    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        logger.debug("Starting forward pass...")

        miner_uids = [
            uid for uid in range(len(self.metagraph.S))
            if check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        ]
        logger.info(f"Miner UIDs: {miner_uids}")

        if len(miner_uids) == 0:
            logger.info("No miners available to query. Exiting forward pass...")
            return

        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        logger.info(f"Current step={self.step}...")

        current_dir = Path.cwd()
        # TODO: Make this dynamic
        repo = "mwaskom/seaborn"
        validator_llm = "gpt4omini"

        ingestion_heuristics = IngestionHeuristics(
            min_files_to_consider_dir_for_problems=3,
            min_file_content_len=50,
        )

        author_name, repo_name = repo.split("/")

        logger.info(f"Cloning repo {repo}...")
        local_repo_dir = clone_repo(author_name, repo_name, current_dir.parent)
        logger.info(f"Finished cloning repo {repo}")

        num_problems_to_gen = 1
        problems: List[GeneratedProblemStatement] = create_problem_statements(
            validator_llm, repo, local_repo_dir, num_problems_to_gen, ingestion_heuristics
        )
        problem: GeneratedProblemStatement = problems[0]
        logger.info(f"Problem statement is: {problem.problem_statement[:50]}...")

        # todo: create proper task ID
        task_id = f"{repo}-{problem.problem_statement[:10]}"

        logger.info(f"Sending task {task_id} to miners, ...")
        responses: List[CodingTask] = await self.dendrite(
            axons=axons,
            synapse=CodingTask(
                repo=repo,
                problem_statement=problem.problem_statement,
                patch=None,
            ),
            deserialize=False,
            timeout=timedelta(minutes=CODINGTASK_TIMEOUT_MINS).total_seconds(), # TODO: need a better timeout method
        )
        logger.info(f"Received patches from miners for task {task_id}: "
                    f"{[(r.patch[:100] + '...' if r.patch else r.patch) for r in responses]}")

        working_miner_uids: List[int] = []
        finished_responses: List[IssueSolution] = []
        process_times: List[float] = []

        logger.info("Checking which received patches are valid...")
        for response in responses:
            if not response:
                logger.info(f"Miner with hotkey {response.axon.hotkey} did not give a response")
            elif response.patch in [None, ""] or not response.axon or not response.axon.hotkey:
                logger.info(f"Miner with hotkey {response.axon.hotkey} gave a response object but no patch")
            else:
                logger.info(f"Miner with hotkey {response.axon.hotkey} gave a valid response/patch")
                uid = next(uid for uid, axon in zip(miner_uids, axons) if axon.hotkey == response.axon.hotkey)
                working_miner_uids.append(uid)
                finished_responses.append(IssueSolution(response.patch))
                process_times.append(response.dendrite.process_time)

        if len(working_miner_uids) == 0:
            logger.info("No miners responded. Exiting forward pass...")
            return
        
        # TODO: Add punishment for miners who did not respond

        logger.info(f"Running task-specific handlers for {task_id}")
        await self.handle_synthetic_patch_response(
            repo,
            problem,
            finished_responses, 
            process_times,
            working_miner_uids,
        )


    async def handle_synthetic_patch_response(
        self,
        repo: str,
        problem: GeneratedProblemStatement,
        finished_responses: List[IssueSolution],
        process_times: List[float], 
        working_miner_uids: List[int], 
    ) -> None:
        try:
            rewards_list = await Validator.calculate_rewards(
                repo,
                problem,
                finished_responses, 
                process_times,
            )
        except Exception:
            logger.exception("Error calculating rewards")
            return

        logger.info(f"Rewards: {rewards_list}")

        # reward the miners who succeeded
        self.update_scores(
            rewards_list,
            working_miner_uids,
            TaskType.LABELLED_ISSUE
        )

        # Crashing due to error: "AttributeError: 'numpy.float32' object has no attribute 'hotkey'"
        # try:
        #     await self.upload_solution(
        #         problem.problem_statement,
        #         finished_responses,
        #         rewards_list.tolist(),
        #         [self.metagraph.S[uid].hotkey for uid in working_miner_uids],
        #     )
        # except Exception:
        #     logger.exception("Error uploading solution")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            time.sleep(5)
