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

import logging
import os
from pathlib import Path
import subprocess
from datetime import datetime
from datetime import timedelta
from typing import List, Optional, Union, Tuple

import pytz
import time
from github.PullRequest import PullRequest
from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv

import numpy as np
import requests
from aiohttp import BasicAuth, ClientSession
from neurons.classes import PendingRewards, LabelledIssueTask, OpenIssueTask
from neurons.constants import DATA_ENDPOINT_BY_TASK, UPLOAD_ISSUE_ENDPOINT, REGISTER_PR_ENDPOINT, \
    PENDING_REWARDS_ENDPOINT, NO_MINER_RESPONSE_SCORE
from taogod.base.validator import BaseValidatorNeuron, TaskType
from taogod.code_compare import compare_and_score
from taogod.protocol import CodingTask
from taogod.s3_utils import download_repo_locally
from taogod.utils.uids import check_uid_availability


# Custom formatter to include line number and PST time
class ESTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        est = pytz.timezone("America/New_York")
        ct = datetime.fromtimestamp(record.created, est)
        return ct.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        # Pad the level name to 7 characters
        record.levelname = f"{record.levelname:<5}"
        return super().format(record)

logging.getLogger().handlers.clear()  # Remove any existing handlers

logger = logging.getLogger(__name__)

# Create and set the handler with the custom formatter
handler = logging.StreamHandler()
handler.setFormatter(ESTFormatter('%(asctime)s - %(filename)s:%(lineno)d [%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


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
    async def calculate_rewards(
        challenge: LabelledIssueTask, 
        responses: List[str],
        local_code_path: Path
    ) -> np.ndarray:
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
        response_scores: np.ndarray,
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
        except Exception:
            logger.exception("Error uploading closed issue")

    @staticmethod
    async def open_pr(patch: str, open_issue: OpenIssueTask, keypair, miner_hotkey) -> Optional[PullRequest]:
        """
        Open a PR with the given patch and open issue task.
        """
        logger.info(f"Entering open_pr with repo URL: {open_issue.repo_url}, issue URL: {open_issue.issue_url}, base commit: {open_issue.base_commit}")
        logger.debug(f"open_issue: {repr(open_issue)}")

        try:

            logger.debug("Setting up sweagent to create a PR...")
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
            logger.debug(f"Saving patch to {path_to_patch}...")
            with open(path_to_patch, "w") as f:
                f.write(patch)
            logger.debug(f"Patch saved to {path_to_patch}")

            docker_path_to_patch = "/root/model.patch"
            logger.debug(f"Copying local {path_to_patch} file to {docker_path_to_patch} in Docker container...")
            subprocess.run(
                f"docker cp {path_to_patch} {env.container_name}:{docker_path_to_patch}",
                shell=True,
                check=False,
            )
            logger.debug(f"Finished copying local {path_to_patch} file to {docker_path_to_patch} in Docker container")

            logger.debug(f"Applying git patch at Docker {docker_path_to_patch}...")
            env.communicate_with_handling(
                input=f"git apply {docker_path_to_patch}",
                error_msg="Failed to apply test patch correctly",
            )

            logger.debug(f"Removing local {path_to_patch}...")
            os.remove(path_to_patch)

            logger.info("Creating Github PR...")
            pr = env.open_pr(trajectory="", _dry_run=False)
            if not pr:
                logger.error(f"Failed to create Github PR. pr object was not successfully generated by env.open_pr")
                return
        except Exception:
            logger.exception("Failed to create Github PR, exception thrown by env.open_pr")
            return

        logger.info(f"Github PR created. Registering it with data endpoint at {REGISTER_PR_ENDPOINT}...")
        try:
            hotkey = keypair.ss58_address
            signature = f"0x{keypair.sign(hotkey).hex()}"
            async with ClientSession() as session:
                payload = {
                    "pr_url": pr.html_url,
                    "miner_hotkey": miner_hotkey
                }
                logger.debug(f"Payload being sent to {REGISTER_PR_ENDPOINT}: {payload}")
                logger.info("Sending request to register PR...")
                async with session.post(
                    url=REGISTER_PR_ENDPOINT,
                    auth=BasicAuth(hotkey, signature),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    logger.info(f"Response received from {REGISTER_PR_ENDPOINT}: {json_response}")
        except Exception:
            logger.exception("Error registering PR")
            # TODO: Handle error casez

    async def assign_pending_rewards_scores(self, miner_uids: List[int]) -> None:
        logger.info(f"Entering pending rewards assignment. Making request to {PENDING_REWARDS_ENDPOINT} ...")
        try:
            async with ClientSession() as session:
                async with session.get(PENDING_REWARDS_ENDPOINT) as raw_response:
                    logger.info(f"Received response from {PENDING_REWARDS_ENDPOINT} ...")

                    raw_response.raise_for_status()
                    response = await raw_response.json()
                    logger.info(f"Parsed response from {PENDING_REWARDS_ENDPOINT}")
                    logger.debug(f"Response is: {response}")

                    pending_rewards = [
                        PendingRewards.model_validate(reward)
                        for reward in response
                    ]
                    logger.debug(f"pending_rewards: {pending_rewards}")
                    if not pending_rewards:
                        logger.info("No pending rewards available, skipping assignment...")
                        return

                    # Give weights to the miners who opened PRs that got accepted
                    rewards = np.ones_like(len(pending_rewards))
                    logger.debug(f"rewards: {rewards}")
                    uids = [reward.uid for reward in pending_rewards]
                    logger.debug(f"uids: {uids}")

                    logger.debug("Calling update_scores...")
                    self.update_scores(rewards, uids, TaskType.OPEN_ISSUE)
                    logger.debug("Finished update_scores")

                    # Set score to 0 for miners who don't have open PR
                    bad_miner_uids = [uid for uid in miner_uids if uid not in uids]
                    logger.info(f"Miners without open PRs: {bad_miner_uids}")
                    penalty_tensor = np.zeros_like(bad_miner_uids)
                    self.update_scores(penalty_tensor, bad_miner_uids, TaskType.OPEN_ISSUE)

        except Exception:
            logger.exception("Error fetching pending rewards")
            return
        logger.info("Finished pending rewards assignment")

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

        await self.assign_pending_rewards_scores(miner_uids)

        # TODO: check if state is actually getting updated
        task_types = [LabelledIssueTask] if self.step < 50 else [LabelledIssueTask, OpenIssueTask]
        logger.info(f"Current step={self.step}, tasks that will be assigned are: {[t.__name__ for t in task_types]}...")

        code_challenge: Optional[Union[LabelledIssueTask, OpenIssueTask]] = None
        for task_type in task_types:
            try:
                logger.info(f"Fetching {task_type.__name__} from {DATA_ENDPOINT_BY_TASK[task_type]} ...")
                response = requests.get(DATA_ENDPOINT_BY_TASK[task_type]).json()
                logger.info(f"Unparsed response keys: {response.keys()}")

                logger.info(f"Fetched {task_type.__name__} from {DATA_ENDPOINT_BY_TASK[task_type]} ."
                            f" Parsing task...")

                code_challenge = task_type.model_validate(response)

                if task_type == LabelledIssueTask:
                    jobs_dir = Path("jobs")
                    jobs_dir.mkdir(exist_ok=True, parents=True)
                    local_code_path = download_repo_locally(code_challenge.s3_repo_url, jobs_dir)
                    logger.info(f"Parsed LabelledIssueTask. S3 url: {code_challenge.s3_repo_url}")
                elif task_type == OpenIssueTask:
                    logger.info(f"Parsed OpenIssueTask. "
                                f"Repo url: {code_challenge.repo_url}, issue url: {code_challenge.issue_url}")

            except Exception:
                logger.exception(f"Error fetching {task_type.__name__} from {DATA_ENDPOINT_BY_TASK[task_type]}. "
                                 f"Exiting forward pass...")
                return

            logger.info(f"Sending task {code_challenge.s3_repo_url} to miners, ...")
            responses: List[CodingTask] = await self.dendrite(
                axons=axons,
                synapse=CodingTask(
                    problem_statement=code_challenge.problem_statement,
                    s3_code_link=code_challenge.s3_repo_url,
                    patch=None,
                ),
                deserialize=False,
                timeout=timedelta(minutes=30).total_seconds(), # TODO: need a better timeout method
            )
            logger.info(f"Received patches from miners for task {code_challenge.s3_repo_url}: "
                        f"{[(r.patch[:100] + '...' if r.patch else r.patch) for r in responses]}")

            working_miner_uids: List[int] = []
            finished_responses: List[str] = []

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
                    finished_responses.append(response.patch)

            if len(working_miner_uids) == 0:
                logger.info("No miners responded. Exiting forward pass...")
                return

            logger.info(f"Running task-specific handlers for {task_type.__name__}")
            if task_type == LabelledIssueTask:
                await self.handle_labelled_issue_response(code_challenge, finished_responses, working_miner_uids)
            elif task_type == OpenIssueTask:
                await self.handle_open_issue_response(code_challenge, finished_responses, working_miner_uids)

            # update scores for miners who failed
            # give min reward to miners who didn't respond
            bad_miner_uids = [uid for uid in miner_uids if uid not in working_miner_uids]
            penalty_tensor = np.array([NO_MINER_RESPONSE_SCORE] * len(bad_miner_uids))
            logger.info(f"Bad miner UIDs: {bad_miner_uids}")
            self.update_scores(
                penalty_tensor,
                bad_miner_uids,
                TaskType.LABELLED_ISSUE if task_type == LabelledIssueTask else TaskType.OPEN_ISSUE
            )

            logger.info(f"Finished forward pass for {task_type.__name__}")

        logger.info(f"Finished forward pass for all available task types ({[t.__name__ for t in task_types]})")


    async def handle_open_issue_response(
        self, code_challenge: OpenIssueTask, finished_responses: List[str], working_miner_uids: List[int]
    ) -> None:
        logger.debug("Entering handle_open_issue response...")
        best_patch, best_uid = Validator.get_best_submission(
            self.scores, 
            working_miner_uids, 
            finished_responses
        )

        if best_patch is None or best_uid is None:
            logger.error("best patch is None. Skipping opening PR.")
            return
        elif self.scores[best_uid] < 0.9:
            logger.info("No good PRs found")
            return

        await Validator.open_pr(best_patch, code_challenge, self.dendrite.keypair, self.metagraph.hotkeys[best_uid])
        logger.debug("Exiting handle_open_issue response")


    async def handle_labelled_issue_response(
        self, 
        code_challenge: LabelledIssueTask, 
        finished_responses: List[str], 
        working_miner_uids: List[int]
    ) -> None:
        try:
            local_code_path = ...
            rewards_list = await Validator.calculate_rewards(
                code_challenge, 
                finished_responses,
                local_code_path
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

        # Upload the closed issue to the data endpoint
        try:
            await self.upload_closed_issue(
                code_challenge,
                finished_responses,
                rewards_list,
                [self.metagraph.hotkeys[uid] for uid in working_miner_uids],
            )
        except Exception:
            logger.exception("Error uploading closed issue")


    @staticmethod
    def get_best_submission(
        scores: np.ndarray, working_miner_uids: List[int], finished_responses: List[str]
    ) -> Tuple[str, int]:
        logger.debug(f"Getting best submission. scores={scores}, "
                     f"working_miner_uids={working_miner_uids}, "
                     f"len(finished_responses)=${len(finished_responses)}")
        uid_to_response = dict(zip(working_miner_uids, finished_responses))
        working_scores = np.array([scores[uid] for uid in working_miner_uids])

        best_uid = working_miner_uids[np.argmax(working_scores)]
        best_patch = uid_to_response[best_uid]

        logger.debug(f"The best miner uid is {best_uid}, with score {np.max(working_scores)}. "
                     f"The best patch is {len(best_patch)} chars long")
        logger.debug("Exiting get_best_submission")
        return best_patch, best_uid


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            time.sleep(5)
