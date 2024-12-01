import os
import subprocess
from pathlib import Path
from typing import Final

import openai
import unidiff
from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv
from sweagent.environment.utils import PatchFormatter

from neurons.helpers import logger

# Todo: replace this with corcel impl
OPENAI_CLIENT: Final[openai.Client] = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

def new_compare(problem_statement: str, patch: str, codebase: Path) -> float:
    # First check if its a valid diff
    try:
        diff = unidiff.PatchSet(patch)
    except Exception as e:
        logger.exception(f"Error during unidiff.PatchSet: {e}")
        return 0.0

    if len(diff) == 0:
        logger.info("No changes in the patch. Returning 0...")
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
        logger.exception("Failed to apply patch, returning 0...")
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
        - Assess the patch for correctness, completeness, and effectiveness in solving the problem.
        - Consider any potential side effects or issues introduced by the patch.
        - Grade a concise solution higher than a lengthy one assuming both are correct and complete.
        - Provide a numerical score out of 100 representing how well the patch solves the problem:
        - 100 means the patch perfectly and completely solves the problem.
        - 0 means the patch does not address the problem at all.
        - If you do not know for sure that the patch perfectly and completely solved the problem, do not give it 100. Instead, give it some value between 0 and 100. Be harshly critical of the submissions you receive, think carefully to find ways in which they may have issues, and make sure the score is reduced appropriately. You will be penalized more harshly if you give scores that are too high than scores that are too low, so bias on the side of giving lower scores.
    
    Output and object of the following format:
    {{
        "explanation": <explanation of your reasoning>   
        "score": <integer of the score>,
    }}
    
    Put all explanation in the "explanation" key of this object. In the "score" key, put only and integer of the score. The explanation should give a clear rationale for why you assigned the score you did. 
    Ouput this object only, in valid JSON format. DO NOT output any additional text or context, like backticks (```) or anything like that.

    Problem Statement: {problem_statement}
    patch: {patch}
    Affected Files:
    {files_affected}
    """

    try:
        logger.info(f"Making OpenAI call with prompt...")
        response = OPENAI_CLIENT.chat.completions.create(
            model='o1-preview-2024-09-12',
            messages=[{"role": "user", "content": prompt}],
        )
        logger.info(f"response: {response}")
        output = response.choices[0].message.content

        logger.info(f"output is {output}")
        response_obj = eval(output)
        score = int(response_obj["score"])
    except Exception as e:
        logger.exception(f"Error during OpenAI API call for new_compare: {e}")
        return 0.0

    return score/100.0
