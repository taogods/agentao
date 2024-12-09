from typing import List
from pathlib import Path
import numpy as np
from sweagent.environment.swe_env import SWEEnv, EnvironmentArguments
from taogod.synthetic_testing import apply_patch, compare_test_results, run_tests
from concurrent.futures import ThreadPoolExecutor

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
        apply_patch(env, test_patch)
        apply_patch(env, response)

        # Run tests after applying patches
        tests_after = run_tests(env)

        # Compare test results
        results = compare_test_results(tests_before, tests_after)
        return results
    except Exception as e:
        print(f"Error in synthetic rewards: {e}")
        return None

def calculate_rewards(
    responses: List[str],
    codebase: Path,
    test_patch: str,
) -> np.ndarray:
    """
    Validate the responses from the miners. This function should score the responses and return a list of rewards for each miner.

    Args:
        responses (List[str]): The responses from the miners.
        codebase (Path): The path to the codebase.
        test_patch (str): The test patch to apply.
    """
    # Initialize the environment
    env_args = EnvironmentArguments(
        image_name="sweagent/swe-agent:latest",
        data_path="text://example.json",  # Doesn't matter for tests
        repo_path=str(codebase),
        verbose=True,
        environment_setup=str('SWE-agent/config/environment_setup/py310_default.yaml'),
    )
    env = SWEEnv(env_args)
    env.reset(0)

    # Run initial tests before applying patches
    tests_before = run_tests(env)

    # Share `tests_before` and other data across processes by making them part of the input arguments
    tasks = [(response, env_args, test_patch, tests_before) for response in responses]

    with ThreadPoolExecutor() as executor:
        synthetic_tests = list(executor.map(process_response_wrapper, tasks))

    # Calculate synthetic rewards
    syn_tests_arr = np.array([])
    for test in synthetic_tests:
        if test is None:
            syn_tests_arr = np.append(syn_tests_arr, 0.0)
        else:
            syn_tests_arr = np.append(
                syn_tests_arr,
                2 * int(len(test["PASS_TO_FAIL"]) == 0)
                + int(len(test["FAIL_TO_PASS"]) >= 0)
                + 3 * int(len(test["NEW_FAIL"]) == 0)
            )

    return syn_tests_arr

if __name__ == "__main__":
    REPO_PATH = Path("mwaskom-seaborn-ae7acf0ce6e30ae773f513e0ccadbb7341cf5e90")

    # Load submission.txt file
    with open('SWE-agent/submission.txt') as f:
        submission = f.read()

    with open('SWE-agent/submission_test.txt') as f:
        submission_test = f.read()

    print(submission)
    # Calculate rewards
    rewards = calculate_rewards(
        [submission]*3, 
        REPO_PATH, 
        submission_test
    )

    print(rewards)