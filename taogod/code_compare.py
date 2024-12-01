import json
import os
from statistics import mean
import subprocess
from typing import Dict, Final

import bittensor as bt
import openai
import sys
import unidiff
from pathlib import Path

from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv
from sweagent.environment.utils import PatchFormatter

# Todo: replace this with corcel impl
OPENAI_CLIENT: Final[openai.Client] = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))


def extract_requirements(issue_text):
    prompt = f"""You are an assistant that extracts key requirements and expected behaviors from issue descriptions.

    Issue Description:
    {issue_text}

    Please provide a concise, bulleted list of the key requirements and expected behaviors extracted from the above issue description."""
        
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{"role": "user", "content": prompt}],
        )
        requirements = response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during OpenAI API call for extract_requirements: {e}")
        sys.exit(1)
    return requirements

def analyze_patch(patch_text):
    prompt = f"""You are an assistant that analyzes code patches and provides a summary of the changes.

    Patch:
    {patch_text}

    Please provide a concise summary of the changes made in the above patch, including:
    - The files and functions affected.
    - The modifications made.
    - The intended effect of these changes."""
    
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during OpenAI API call for analyze_patch: {e}")
        sys.exit(1)
    return analysis

def compare_patch_to_requirements(patch_analysis, requirements):
    prompt = f"""You are an assistant that compares patch changes to issue requirements.

    Issue Requirements:
    {requirements}

    Patch Analysis:
    {patch_analysis}

    Based on the above, please determine to what extent the patch addresses the issue requirements. Provide a summary of which requirements are met, which are partially met, and which are not met."""
    
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{"role": "user", "content": prompt}],
        )
        comparison = response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during OpenAI API call for compare_patch_to_requirements: {e}")
        sys.exit(1)
    return comparison

def compare_patches(patch_analysis1: str, patch_analysis2: str) -> Dict[str, int]:
    prompt = f"""You are an assistant that compares two code patches.

    Patch Analysis 1:
    {patch_analysis1}

    Patch Analysis 2:
    {patch_analysis2}

    Please compare the two patches and determine their similarity on the following metrics:
    1. issue_similarity (0-100): How similar are the issues being addressed?
    2. files_similarity (0-100): How similar are the files being modified?
    3. functions_similarity (0-100): How similar are the functions being affected?
    4. logic_similarity (0-100): How similar are the logical changes being made?
    5. overall_similarity (0-100): What is the overall similarity of the patches?

    Provide your analysis as a JSON object with these exact keys: [
        "issue_similarity", 
        "files_similarity", 
        "functions_similarity", 
        "logic_similarity", 
        "overall_similarity"
    ]. The values should be integers between 0 and 100. 
    Do not include any special formatting like ```json, etc. 
    The output should be formatted only as a parseable JSON dict
    """
    
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        comparison =  response.choices[0].message.content

        # Parse the JSON string to a Python dictionary
        comparison_dict = json.loads(comparison)
    except Exception as e:
        print(f"Error during OpenAI API call or JSON parsing for compare_patches: {e}")
        return {"overall_similarity": 100}  # TODO: Handle better

    return comparison_dict


def compare_and_score(gt_patch, miner_patch) -> float:
    """
    Conducts LLM comparison between the ground truth patch and the miner's 
    patch and returns a score between 0 and 1.
    """
    # Compare the miner's patch to the ground truth patch
    comparison: Dict = compare_patches(gt_patch, miner_patch)

    bt.logging.info(f"Comparison results: {comparison}")
    score = mean(comparison.values()) / 100
    return score

def new_compare(problem_statement: str, patch: str, codebase: Path) -> float:
    # First check if its a valid diff
    try:
        diff = unidiff.PatchSet(patch)
    except Exception as e:
        print(f"Error during unidiff.PatchSet: {e}")
        return 0.0

    if len(diff) == 0:
        print("No changes in the patch")
        return 0.0

    # Apply the patch against the codebase to see if it works
    env_args = EnvironmentArguments(
                image_name="sweagent/swe-agent:latest",
                data_path=f"text://{problem_statement}",
                repo_path=str(codebase),
                verbose=True,
            )
    env = SWEEnv(env_args)
    env.reset()
    try:
        path_to_patch = "model.patch"
        with open(path_to_patch, "w") as f:
            f.write(patch)

        docker_path_to_patch = "/root/model.patch"
        subprocess.run(
            f"docker cp {path_to_patch} {env.container_name}:{docker_path_to_patch}",
            shell=True,
            check=False,
        )

        env.communicate_with_handling(
            input=f"git apply {docker_path_to_patch}",
            error_msg="Failed to apply test patch correctly",
        )

        os.remove(path_to_patch)
    except Exception as e:
        return 0.0

    def read_file(path: str) -> str:
        with open(codebase / path, "r") as file:
            return file.read()

    patch_formatter = PatchFormatter(patch, read_file)
    files_affected = patch_formatter.concat_files_strings(patch_formatter._patched_files)

    prompt = f"""
    Instructions:
    You are tasked with evaluating a code patch to determine how well it addresses a specific problem. Please follow these steps:
    Read the Problem Statement to understand the issue that needs to be resolved.
    Review the Git Diff to see the changes introduced by the patch.
    Examine the Affected Files to understand the context of the changes.
    Your Task:
    Assess the patch for correctness, completeness, and effectiveness in solving the problem.
    Consider any potential side effects or issues introduced by the patch.
    Grade a concise solution higher than a lengthy one assuming both are correct and complete.
    Provide a numerical score out of 100 representing how well the patch solves the problem:
    100 means the patch perfectly and completely solves the problem.
    0 means the patch does not address the problem at all.
    Please output only the scalar score (an integer between 0 and 100). DO NOT
    output any additional text or context.
    Problem Statement: {problem_statement}
    patch: {patch}
    Affected Files:
    {files_affected}
    """

    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model='o1-preview-2024-09-12',
            messages=[{"role": "user", "content": prompt}],
        )
        score = response.choices[0].message.content
        score = int(score)
    except Exception as e:
        print(f"Error during OpenAI API call for new_compare: {e}")
        return 0.0

    return score/100.0

if __name__ == "__main__":
    compare_and_score("", "")