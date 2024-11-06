# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Taoception

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


import bittensor as bt
import json
import numpy as np
import os
import random
import requests
import string
import subprocess
import time
import typing
# Bittensor
from aiohttp import BasicAuth, ClientSession
from github.PullRequest import PullRequest
from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv
from typing import Final, List, Optional

from neurons.classes import LabelledIssueTask, OpenIssueTask, PendingRewards
# import base validator class which takes care of most of the boilerplate
from taoception.base.validator import BaseValidatorNeuron, TaskType
# Bittensor Validator Template:
from taoception.code_compare import compare_and_score
from taoception.protocol import CodingTask
from taoception.utils.uids import check_uid_availability

NO_RESPONSE_MINIMUM: float = 0.005
ISSUES_DATA_ENDPOINT: Final[str] = "https://gh-issue-pull.onrender.com/task"
OPEN_ISSUE_ENDPOINT: Final[str] = "https://gh-issue-pull.onrender.com/open_issue"
PENDING_REWARDS_ENDPOINT: Final[str] = "https://gh-issue-pull.onrender.com/pending_rewards"
REGISTER_PR_ENDPOINT: Final[str] = "https://gh-issue-pull.onrender.com/register_pr"
UPLOAD_ISSUE_ENDPOINT: Final[str] = "https://gh-issue-pull.onrender.com/upload_issue"
DOCKER_CACHE_LEVEL: str = "instance"

def generate_random_string(length: int) -> str:
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        # TODO(developer): Anything specific to your use case you can do here

    async def calculate_rewards(self, challenge: LabelledIssueTask, responses: typing.List[str]) -> np.ndarray:
        """
        Validate the responses from the miners. This function should score the responses and return a list of rewards for each miner.
        """
        return np.array([
            compare_and_score(challenge.patch, response)
            for response in responses
        ])
    
    async def upload_closed_issue(
        self, 
        issue: LabelledIssueTask, 
        response_patches: List[str],
        response_scores: List[float],
        miner_hotkeys: List[str],
    ) -> None:
        """
        Upload the closed issue to the data endpoint.
        """
        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"
        try:
            async with ClientSession() as session:
                # TODO: Add how long it takes to upload the issue
                payload = [{
                    "problem_statement": issue.problem_statement,
                    "solution_patch": response_patch,
                    "score": response_score,
                    "miner_hotkey": miner_hotkey,
                } for 
                    response_patch, 
                    response_score, 
                    miner_hotkey 
                    in zip(response_patches, response_scores, miner_hotkeys)
                ]
                async with session.post(
                    url=UPLOAD_ISSUE_ENDPOINT, 
                    auth=BasicAuth(hotkey, signature),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    _result = await response.json()
        except Exception as e:
            bt.logging.error(f"Error uploading closed issue: {e}")
    
    def open_pr(self, patch: str, open_issue: OpenIssueTask) -> Optional[PullRequest]:
        """
        Open a PR with the given patch and open issue task.
        """
        try:
            bt.logging.info(f"{open_issue.issue_url} {open_issue.base_commit}")
            script_arguments = EnvironmentArguments(
                image_name="sweagent/swe-agent:latest",
                data_path=open_issue.issue_url,
                repo_path=open_issue.repo_url,
                base_commit=open_issue.base_commit,
                verbose=True,
            )
            env = SWEEnv(script_arguments)
            env.reset()
            path_to_patch = "model.patch"
            with open(path_to_patch, "w") as f:
                f.write(patch)
            
            subprocess.run(
                f"docker cp {path_to_patch} {env.container_name}:/root/model.patch",
                shell=True,
                check=False,
            )

            env.communicate_with_handling(
                input="git apply /root/model.patch",
                error_msg="Failed to apply test patch correctly",
            )
            os.remove(path_to_patch)
            bt.logging.info("Opening PR")
            pr = env.open_pr(trajectory="", _dry_run=False)
            return pr
        except Exception as e:
            bt.logging.error(f"Error opening PR: {e}")
            return None
        
    async def pending_rewards(self, miner_uids: List[int]) -> None:
        try:
            async with ClientSession() as session:
                async with session.get(PENDING_REWARDS_ENDPOINT) as response:
                    response.raise_for_status()
                    pending_rewards = [
                        PendingRewards.model_validate(reward) 
                        for reward in response.json()
                    ]
                    if pending_rewards == []:
                        return

                    # Give weights to the miners who opened PRs that got accepted
                    rewards = np.ones_like(len(pending_rewards))
                    uids = [reward.uid for reward in pending_rewards]
                    self.update_scores(rewards, uids, TaskType.OPEN_ISSUE)

                    # Set score to 0 for miners who don't have open PR
                    bad_miner_uids = [uid for uid in miner_uids if uid not in uids]
                    penalty_tensor = np.zeros_like(bad_miner_uids)
                    self.update_scores(penalty_tensor, bad_miner_uids, TaskType.OPEN_ISSUE) 
        except Exception as e:
            bt.logging.error(f"Error fetching pending rewards: {e}")
            return

    async def closed_issue(self, miner_uids: List[int]) -> None:
        try:
            response = requests.get(ISSUES_DATA_ENDPOINT).json()
            code_challenge: LabelledIssueTask = LabelledIssueTask.model_validate(response)
        except Exception as e:
            bt.logging.error(f"Error fetching issue from data endpoint: {e}. Skipping forward pass")
            return

        bt.logging.debug(f"Received response from data endpoint: {response.keys()}")

        # The dendrite client queries the network.
        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        responses: typing.List[CodingTask] = await self.dendrite(
            axons=axons,
            synapse=CodingTask(
                problem_statement=code_challenge.problem_statement,
                s3_code_link=code_challenge.s3_repo_url,
                patch=None,
            ),
            deserialize=False,
            timeout=6000, # TODO: need a better timeout method
        )

        bt.logging.info(f"Received patches: {[r.patch for r in responses]}")

        working_miner_uids = []
        finished_responses = []

        for response in responses:
            if not response:
                bt.logging.info("No response from miner")
            elif response.patch in [None, ""] or not response.axon or not response.axon.hotkey:
                bt.logging.info("No patch from miner")
            else:
                uid = next(uid for uid, axon in zip(miner_uids, axons) if axon.hotkey == response.axon.hotkey)
                working_miner_uids.append(uid)
                finished_responses.append(response.patch)

        if len(working_miner_uids) == 0:
            bt.logging.info("No miners responded")
            return

        try:
            rewards_list = await self.calculate_rewards(code_challenge, finished_responses)
        except Exception as e:
            bt.logging.error(f"Error calculating rewards: {e}")
            return

        bt.logging.debug(f"Rewards: {rewards_list}")

        # reward the miners who succeeded
        self.update_scores(
            rewards_list, 
            working_miner_uids, 
            TaskType.LABELLED_ISSUE
        )

        # Upload the closed issue to the data endpoint
        try:
            await self.upload_closed_issue(
                code_challenge, 
                finished_responses, 
                rewards_list,
                [self.metagraph.hotkeys[uid] for uid in working_miner_uids],
            )
        except Exception as e:
            bt.logging.error(f"Error uploading closed issue: {e}")

        # update scores for miners who failed
        # give min reward to miners who didn't respond
        bad_miner_uids = [uid for uid in miner_uids if uid not in working_miner_uids]
        penalty_tensor = np.array([NO_RESPONSE_MINIMUM] * len(bad_miner_uids))
        bt.logging.debug(f"Bad miner UIDs: {bad_miner_uids}")
        self.update_scores(penalty_tensor, bad_miner_uids, TaskType.LABELLED_ISSUE)

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # get all the miner UIDs
        miner_uids = [
            uid for uid in range(len(self.metagraph.S))
            if check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        ]
        if len(miner_uids) == 0:
            bt.logging.info("No miners available to query.")
            return
        
        bt.logging.info(f"Miner UIDs: {miner_uids}")
        
        # The dendrite client queries the network.
        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        # ================================================================
        # ======================= Pending Rewards ========================
        # ================================================================
        # This section rewards miners who have opened PRs that have been 
        # accepted.

        self.pending_rewards(miner_uids)

        # =============================================================
        # ======================== Coding Task ========================
        # =============================================================
        # This section contains the logic for the closed issues loop

        self.closed_issue(miner_uids)

        # ================================================================
        # ========================= Open PR Task =========================
        # ================================================================
        # This sections contains the logic for the open issues loop

        if self.step < 50: return # Build up weights first

        # ping for open issue challenges
        # Generate a coding problem for the miners to solve.
        try:
            response = requests.get(OPEN_ISSUE_ENDPOINT).json()
            code_challenge: OpenIssueTask = OpenIssueTask.model_validate(response)
        except requests.exceptions.HTTPError as error:
            bt.logging.error(f"Error fetching issue from data endpoint: {error}. Skipping forward pass")
            return
        except Exception as e:
            bt.logging.error(f"Error fetching issue from data endpoint: {e}. Skipping forward pass")
            return
        
        bt.logging.debug(f"Received response from data endpoint: {response.keys()}")

        # Rate them and select the highest rated one that is above a threshold score
        responses = await self.dendrite(
            axons=axons,
            synapse=CodingTask(
                problem_statement=code_challenge.problem_statement,
                s3_code_link=code_challenge.s3_repo_url,
                patch=None,
            ),
            deserialize=False,
            timeout=600,
        )

        working_miner_uids = []
        finished_responses = []

        for response in responses:
            if not response:
                bt.logging.info("No response from miner")
                continue
            if response.patch in [None, ""] or not response.axon or not response.axon.hotkey:
                continue
            
            uid = [uid for uid, axon in zip(miner_uids, axons) if axon.hotkey == response.axon.hotkey][0]
            working_miner_uids.append(uid)
            finished_responses.append(response.patch)

        if len(working_miner_uids) == 0:
            bt.logging.info("No miners responded")
            return

        # Take the response from the highest rated miner in self.scores
        uid_to_response = {uid:response 
                           for uid, response 
                           in zip(working_miner_uids, finished_responses)
                        }
        working_scores = np.array([self.scores[uid] for uid in working_miner_uids])

        best_response_uid = working_miner_uids[np.argmax(working_scores)]
        best_response = uid_to_response[best_response_uid]

        assert best_response is not None

        # If its good, open a PR with the patch
        if self.scores[best_response_uid] < 0.9:
            bt.logging.info("No good PRs found")
            return

        # open a PR with the patch
        pr = self.open_pr(best_response, code_challenge)

        # Register PR with backend
        if pr:
            keypair = self.dendrite.keypair
            hotkey = keypair.ss58_address
            signature = f"0x{keypair.sign(hotkey).hex()}"
            try:
                async with ClientSession() as session:
                    payload = {
                        "pr_url": pr.html_url,
                        "miner_hotkey": self.metagraph.hotkeys[best_response_uid]
                    }
                    async with session.post(
                        url=REGISTER_PR_ENDPOINT, 
                        auth=BasicAuth(hotkey, signature),
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        _result = await response.json()
            except Exception as e:
                bt.logging.error(f"Error opening PR: {e}")
                # TODO: Handle error casez


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
