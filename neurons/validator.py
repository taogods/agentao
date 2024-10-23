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
import random
import string
import time
import typing
from typing import Final

# Bittensor
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

from neurons.classes import LabelledIssueTask
# import base validator class which takes care of most of the boilerplate
from taoception.base.validator import BaseValidatorNeuron
# Bittensor Validator Template:
from taoception.protocol import CodingTask
from taoception.utils.uids import check_uid_availability

NO_RESPONSE_MINIMUM: float = 0.005
ISSUES_DATA_ENDPOINT: Final[str] = "TODO"
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

    async def validate(
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

        # Generate a coding problem for the miners to solve.
        try:
            response = requests.get(ISSUES_DATA_ENDPOINT)
        except requests.exceptions.HTTPError as error:
            print(f"Error fetching issue from data endpoint: {error}. Skipping forward pass")
            return

        code_challenge: LabelledIssueTask = LabelledIssueTask.model_validate(response.json())

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
        
        # The dendrite client queries the network.
        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        responses = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=axons,
            # Construct a dummy query. This simply contains a single integer.
            synapse=CodingTask(
                issue_desc=code_challenge.problem_statement,
                code_link=code_challenge.repo,
            ),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=True,
            timeout=600, # TODO: need a better timeout method
        ) 

        working_miner_uids = []
        finished_responses = []

        for response in responses:
            if response.code_solution in [None, ""] or not response.axon or not response.axon.hotkey:
                continue
            
            uid = [uid for uid, axon in zip(miner_uids, axons) if axon.hotkey == response.axon.hotkey][0]
            working_miner_uids.append(uid)
            finished_responses.append(response.code_solution)

        if len(working_miner_uids) == 0:
            bt.logging.info("No miners responded")
            # TODO: distributed weight evenly

        # Score the responses from the different miners
        rewards_list = await self.validate(code_challenge, finished_responses)

        # reward the miners who succeeded
        rewards = []
        reward_uids = []
        for r, r_uid in zip(rewards_list, working_miner_uids):
            if r is not None:
                rewards.append(r)
                reward_uids.append(r_uid)
        rewards = torch.FloatTensor(rewards).to(self.device)
        # TODO: update scores here

        # update scores for miners who failed
        # give min reward to miners who didn't respond
        bad_miner_uids = [uid for uid in miner_uids if uid not in working_miner_uids]
        penalty_tensor = torch.FloatTensor([NO_RESPONSE_MINIMUM] * len(bad_miner_uids)).to(self.device)
        # TODO: self.update_scores(penalty_tensor, bad_miner_uids)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
