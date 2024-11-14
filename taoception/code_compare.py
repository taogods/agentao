import json
import os
from statistics import mean
from typing import Dict, Final

import bittensor as bt
import openai
import sys

from typing import Optional

from neurons.helpers import logger

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


def compare_and_score(gt_patch, miner_patch, event_id: Optional[str]) -> float:
    """
    Conducts LLM comparison between the ground truth patch and the miner's 
    patch and returns a score between 0 and 1.
    """
    # Compare the miner's patch to the ground truth patch
    comparison: Dict = compare_patches(gt_patch, miner_patch)

    bt.logging.info(f"Comparison results: {comparison}")
    
    if event_id is not None:
        logger.info(
            f"Pushing comparison result to Posthog results",
            extra={
                "event_id": event_id,
                "comparison": comparison
            }
        )

    score = mean(comparison.values()) / 100
    return score

if __name__ == "__main__":
    compare_and_score("this is a patch", "this is another patch", "my_event_id")