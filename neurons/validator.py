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


import json
import os
import random
import string
import subprocess
import time
import typing
from typing import Final, Optional
from github.PullRequest import PullRequest

# Bittensor
from aiohttp import BasicAuth, ClientSession
import bittensor as bt
import docker
import requests
import swebench
import swebench.harness
import swebench.harness.docker_build
import swebench.harness.docker_utils
import swebench.harness.run_evaluation
import swebench.harness.test_spec
import torch

from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv
from neurons.classes import LabelledIssueTask, OpenIssueTask, PendingRewards
# import base validator class which takes care of most of the boilerplate
from taoception.base.validator import BaseValidatorNeuron, TaskType
# Bittensor Validator Template:
from taoception.code_compare import compare_and_score
from taoception.protocol import CodingTask
from taoception.utils.uids import check_uid_availability

NO_RESPONSE_MINIMUM: float = 0.005
ISSUES_DATA_ENDPOINT: Final[str] = "https://forty-geese-watch.loca.lt/task"
OPEN_ISSUE_ENDPOINT: Final[str] = "https://forty-geese-watch.loca.lt/open_issue"
PENDING_REWARDS_ENDPOINT: Final[str] = "https://forty-geese-watch.loca.lt/pending_rewards"
REGISTER_PR_ENDPOINT: Final[str] = "https://forty-geese-watch.loca.lt/register_pr"
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

    async def calculate_rewards_simulated(
            self,
            challenge: LabelledIssueTask,
            responses: typing.List[str],
    ) -> torch.FloatTensor:
        """
        Validate the responses from the miners. This function should score the responses and return a list of rewards for each miner.

        Args:
            challenge (LabelledIssueTask): The challenge that was sent to the miners.
            responses (List[str]): The responses from the miners.

        Returns:
            torch.FloatTensor: A list of rewards for each miner.
        """
        client = docker.from_env()
        existing_images = swebench.list_images(client)
        instance = challenge.to_swebench_instance()
        swebench.harness.docker_build.build_env_images(client, [instance])

        # run tests
        test_specs = swebench.harness.test_spec.make_test_spec(instance)
        tests_passed = []

        for prediction in responses:
            prediction = {
                "instance_id": challenge.instance_id,
                "model_patch": prediction,
                "model_name_or_path": "taoception"
            }
            run_id = generate_random_string(5)
            swebench.harness.run_evaluation.run_instance(
                test_specs,
                prediction,
                True,
                False,
                client,
                "WF"
            )

            test_report = swebench.harness.run_evaluation.make_run_report(
                {prediction["instance_id"]: prediction},
                [instance],
                client,
                run_id
            )
            with test_report.open('r') as f:
                test_report = json.load(f)

            if test_report['error_ids']:
                tests_passed.append(0)
            else: tests_passed.append(1)

        swebench.harness.docker_utils.clean_images(client, existing_images, DOCKER_CACHE_LEVEL, False)

        # TODO(developer): Implement your scoring function here
        return torch.FloatTensor(tests_passed)

    async def calculate_rewards(self, challenge: LabelledIssueTask, responses: typing.List[str]) -> torch.FloatTensor:
        """
        Validate the responses from the miners. This function should score the responses and return a list of rewards for each miner.
        """
        return torch.FloatTensor([
            compare_and_score(challenge.patch, response)
            for response in responses
        ])
    
    def open_pr(self, patch: str, open_issue: OpenIssueTask) -> Optional[PullRequest]:
        """
        Open a PR with the given patch and open issue task.
        """
        try:
            script_arguments = EnvironmentArguments(
                image_name="sweagent/swe-agent:latest",
                data_path=f"text://{open_issue.issue_url}",
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
            pr = env.open_pr(trajectory="", _dry_run=False)
            return pr
        except Exception as e:
            bt.logging.error(f"Error opening PR: {e}")
            return None

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # Generate a coding problem for the miners to solve.
        try:
            # Load json at test_issue.json
            with open("neurons/test_issue.json", "r") as f:
                response = json.load(f)
            # response = requests.get(ISSUES_DATA_ENDPOINT)
            code_challenge: LabelledIssueTask = LabelledIssueTask.model_validate(response)
        except requests.exceptions.HTTPError as error:
            bt.logging.error(f"Error fetching issue from data endpoint: {error}. Skipping forward pass")
            return
        except Exception as e:
            bt.logging.error(f"Error fetching issue from data endpoint: {e}. Skipping forward pass")
            return
        
        bt.logging.debug(f"Received response from data endpoint: {response.keys()}")

        # get all the miner UIDs
        miner_uids = []
        for uid in range(len(self.metagraph.S)):
            uid_is_available = check_uid_availability(
                self.metagraph, uid, self.config.neuron.vpermit_tao_limit
            )
            if uid_is_available:
                miner_uids.append(uid)
        if len(miner_uids) == 0:
            bt.logging.info("No miners available to query.")
            return
        
        bt.logging.info(f"Miner UIDs: {miner_uids}")
        
        # The dendrite client queries the network.
        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        # =============================================================
        # ======================== Coding Task ========================
        # =============================================================
        synpase = CodingTask(
            problem_statement=code_challenge.problem_statement,
            s3_code_link=code_challenge.s3_repo_url,
            patch=None,
        )
        
        responses = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=axons,
            # Construct a dummy query. This simply contains a single integer.
            synapse=synpase,
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
            timeout=600, # TODO: need a better timeout method
        ) 

        bt.logging.info(f"Received responses: {responses}")

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
            # return

        try: 
            rewards_list = await self.calculate_rewards(code_challenge, finished_responses)
        except Exception as e:
            bt.logging.error(f"Error calculating rewards: {e}")
            return
        
        bt.logging.debug(f"Rewards: {rewards_list}")

        # reward the miners who succeeded
        rewards = []
        reward_uids = []
        for r, r_uid in zip(rewards_list, working_miner_uids):
            if r is not None:
                rewards.append(r)
                reward_uids.append(r_uid)
        rewards = torch.FloatTensor(rewards).to(self.device)
        self.update_scores(rewards, reward_uids, TaskType.LABELLED_ISSUE)

        # update scores for miners who failed
        # give min reward to miners who didn't respond
        bad_miner_uids = [uid for uid in miner_uids if uid not in working_miner_uids]
        penalty_tensor = torch.FloatTensor([NO_RESPONSE_MINIMUM] * len(bad_miner_uids)).to(self.device)
        bt.logging.debug(f"Bad miner UIDs: {bad_miner_uids}")
        self.update_scores(penalty_tensor, bad_miner_uids, TaskType.LABELLED_ISSUE)

        # ================================================================
        # ========================= Open PR Task =========================
        # ================================================================

        # ping for open issue challenges
        # Generate a coding problem for the miners to solve.
        try:
            response = requests.get(OPEN_ISSUE_ENDPOINT)
            code_challenge: OpenIssueTask = OpenIssueTask.model_validate(response)
        except requests.exceptions.HTTPError as error:
            bt.logging.error(f"Error fetching issue from data endpoint: {error}. Skipping forward pass")
            return
        except Exception as e:
            bt.logging.error(f"Error fetching issue from data endpoint: {e}. Skipping forward pass")
            return
        
        bt.logging.debug(f"Received response from data endpoint: {response.keys()}")

        # Rate them and select the highest rated one that is above a threshold score
        synapse = CodingTask(
            problem_statement=code_challenge.problem_statement,
            s3_code_link=code_challenge.s3_repo_url,
            patch=None,
        )

        responses = await self.dendrite(
            axons=axons,
            synapse=synapse,
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
            # return

        # Take the response from the highest rated miner in self.scores
        sorted_player_ids = sorted(range(len(self.scores)), key=lambda x: self.scores[x], reverse=True)

        best_response = None
        best_response_uid = None
        for miner_uid in sorted_player_ids:
            response = next(((response, uid) for response, uid in zip(finished_responses, working_miner_uids) if uid == miner_uid), None)
            if response:
                best_response, best_response_uid = response
                break
        bt.logging.debug(f"Rewards: {rewards_list}")

        assert best_response is not None

        # ===========  If its good, open a PR with the patch
        if self.scores[best_response_uid] < 0.9:
            bt.logging.info("No good PRs found")
            return

        # open a PR with the patch
        pr = self.open_pr(best_response, code_challenge)

        # Register PR with backend
        if pr:
            keypair = self.dendrite.keypair
            hotkey = keypair.ss58_address
            signature = f"0x{keypair.sign(hotkey).hex}"
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

        # ============ Get all the accepted PR's that you have not yet rewarded. For each one
        # reward the miner who opened the PR by giving them weight on self.open_pr_weight
        response = requests.get(PENDING_REWARDS_ENDPOINT)
        pending_rewards = [PendingRewards.model_validate(reward) for reward in response.json()]
        if pending_rewards == []:
            return

        # TODO: Check disk db for rewards that have been given. Ask Parshant how we does this
        # with omega focus videos

        # Give weights to the miners who opened PRs that got accepted
        rewards = [1 for _ in pending_rewards]
        uids = [reward.uid for reward in pending_rewards]
        self.update_scores(rewards, uids, TaskType.OPEN_ISSUE)

        # Set score to 0 for miners who don't have open PR
        bad_miner_uids = [uid for uid in miner_uids if uid not in uids]
        penalty_tensor = torch.FloatTensor([0] * len(bad_miner_uids)).to(self.device)
        self.update_scores(penalty_tensor, bad_miner_uids, TaskType.OPEN_ISSUE)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
