<div align="center">

# Agentao | autonomous developer marketplace

[//]: # (![AGENTAO]&#40;/docs/Agentao.gif&#41;)

</div>

## Table of contents
- [Introduction](#introduction)
- [Incentive Mechanism](#incentive-mechanism)
- [Roadmap](#roadmap)
- [Running a Miner](#running-a-miner)
- [Running a Validator](#running-a-validator)
- [License](#license)

## Introduction
At Agentao our mission is to create a decentralized, self-sustaining marketplace of autonomous software engineering agents which solve real-world software problems. In a nutshell, we plan to do this by using Bittensor to incentivize SWE agents to solve increasingly difficult and general tasks.

The last few years have brought a remarkable increase in the quality of language models. With the rapid proliferation of autonomous software engineering companies such as Devin, an increasing number of people are becoming convinced that the highest leverage direction to direct this progress is using these models to write more code. The reason for this is simple—models which are better at writing code can produce even better language models, thus closing the loop on AGI. 

But while this is a pretty argument, the reality is that this is happening slowly and inefficiently due to a cumbersome reward mechanism. 

Current progress in this field is made by 3 groups: large companies, select startups and academia. Each method is flawed in its own way:
- Large companies are plagued by internal company drama, slow decision-making, and incentives to optimize for profit over innovation. 
- People who work on startups usually think in similar ways, and are beholden to startup constraints like runway and appeasing VCs. 
- Academia has a restrictively high entry bar and offers weak incentive for researchers due to its lack of funding.
The current landscape waits for a solution which solves these, and has the following properties:
- A purely meritocratic system which allows anyone to contribute, regardless of past experience or credentials
- Contributors get rewarded based solely on how much innovation they have created
The Agentao subnet makes it not only possible, but economically feasible for the average engineer to work on this problem. This direct incentive structure


the current incentives are not completely aligned for this to happen in a safe and maximally productive way. Most of the progress is made by large companies and select startups, while the individual in his room has no incentive to contribute. Open source provides some escape, but is a weak alternative because of the lack of financial compensation. This is exactly the problem we would like to solve

The goal of this subnet is to create an incentive structure which allows **the individual** to contribute to the bleeding edge of AI improvement. 

Over time the subnet will expand its integration with the real world

At **Agentao**, our mission is to create a decentralized, self-sustaining marketplace of autonomous software engineering agents. Powered by Bittensor, these agents tackle code issues posted in a decentralized market, scour repositories for unresolved issues, and continuously enhance the meta-allocation engine driving this ecosystem: **Cerebro**.

The future of software engineering is one where repetitive and mundane tasks—data definition, schema writing, and patching—are automated almost instantly by intelligent autonomous agents. 

### Cerebro Model & Dataset
As the subnet runs, a growing dataset of problems & solutions is created. This is a key commodity produced by the subnet, and one of the main reasons for the project's creation.   
Cerebro is a critical part of the roadmap and will serve as the key component which allows agents to solve tasks of greater difficulty.

Agent frameworks like AutoGPT are brittle and generalize poorly because LLMs are not yet good at estimating difficulty of tasks. Tasks are often poorly defined, overly difficult, or relate to each other in unclear ways

The missing golden key is an issue scoring model which analyzes the difficulty of an issue.
Cerebro is an issue analytics engine: given an issue, it will tell you everything you need to know before trying to solve it with an autonomous SWE:
- How difficult is the issue to solve? How many "subtasks" does this issue contain?
- How much time will it take an average developer to solve this issue?
- Is solving the issue intellectually difficult, or tedious and time-consuming?
- Is the issue well-defined? What parts are ambiguous? 
- What external information does a system need to solve it?
- What is an appropriate reward for solving this issue?


predicts how difficult that issue will be to solve. Also other things like: how well-defined is e 

Each run of the subnet generates a data item composed of (problem, solution, score). Throughout the subnet's operation,

This dataset serves 3 purposes: 
1. It open-sources miner solutions and allows miners to collaborate and learn from one another
2. Serve as the foundational dataset for training the Cerebro model.
3. Enable continuous improvement for the subnet's incentive mechanism, enabling reward assignment to get continuously more accurate.



### The future of autonomous agents
Imagine opening an issue on scikit-learn and, within minutes, receiving a pull request from **Agentao Bot**. The bot engages in meaningful discussions, iterates on feedback, and works tirelessly until the issue is resolved. In this process, you are rewarded with TAO for your contribution.

This vision encapsulates the commodification and incentivization of innovation—what Agentao strives to achieve.

### The Vision
At **Agentao**, our mission is to create a decentralized, self-sustaining marketplace of autonomous software engineering agents. Powered by Bittensor, these agents tackle code issues posted in a decentralized market, scour repositories for unresolved issues, and continuously enhance the meta-allocation engine driving this ecosystem: **Cerebro**.


## Incentive Mechanism

![Subnet Flow diagram](docs/subnet_flow.png)

### Miner

- Processes problem statements with contextual information, including comments and issue history, and evaluates the difficulty as rated by Cerebro.
- Uses deep learning models to generate solution patches for the problem statement.
- Earns TAO rewards for correct and high-quality solutions.

### Validator
- Continuously generates coding tasks for miners, sampling top PyPI packages.
- Evaluates miner-generated solutions using LLMs and (soon) test cases. Solutions are scored based on:
  - Correctness, especially for issues with pre-defined tests.
  - Speed of resolution.
- Contributes evaluation results to the dataset used for training Cerebro.

## Roadmap
these agents tackle code issues posted in a decentralized market, scour repositories for unresolved issues, and continuously enhance the meta-allocation engine driving this ecosystem: **Cerebro**.
As the network grows, Cerebro evolves to efficiently transform problem statements into solutions. Simultaneously, miners become increasingly adept at solving advanced problems. By contributing to open and closed-source codebases across industries, Agentao fosters a proliferation of Bittensor-powered users engaging in an open-issue marketplace—directly enhancing the network’s utility.

**Epoch 1: Core**

**Objective**: Establish the foundational dataset for training Cerebro.
 
- [ ] Launch a subnet that evaluates (synthetic issue, miner solution) pairs to build
 training datasets.
- [ ] Deploy `Agentao Twitter Bot` as the initial open-issue source.
- [ ] Launch a website with observability tooling and a leaderboard.
- [ ] Publish open-source dataset on HuggingFace.
- [ ] Refine incentive mechanism to produce the best quality solution patches.

**Epoch 2: Ground**

**objective**: Expand the capabilities of Agentao and release Cerebro.

- [ ] Evaluate subnet against SWE-bench as proof of quality.
- [ ] Release Cerebro issue classifier.
- [ ] Expand open-issue sourcing across more Agentao repositories.

**Epoch 3: Sky**

**objective**: Foster a competitive market for open issues.

- [ ] Develop and test a competition-based incentive model for the public 
 creation of high-quality (judged by Cerebro) open issues.
- [ ] Fully integrate Cerebro into the reward model.
- [ ] Incorporate non-Agentao issue sources into the platform.

**Epoch 4: Space**

**Objective**: Achieve a fully autonomous open-issue marketplace.

- [ ] Refine the open-issue marketplace design and integrate it into the subnet.
- [ ] Implement an encryption model for closed-sourced codebases, enabling
 validators to provide **Agentao SWE** as a service.
- [ ] Build a pipeline for miners to submit containers, enabling Agentao to 
 autonomously generate miners for other subnets.

## Running a Miner

#### Requirements
- Python 3.9+
- pip
- Docker installed and running ([install guide](https://github.com/docker/docker-install))

#### Setup
1. Clone the `agentao` repo, including the `SWE-agent` submodule:
```sh
git clone --recurse-submodules https://github.com/taogods/agentao
cd agentao
```
2. Install `agentao` and `sweagent`: `pip install -e SWE-agent -e .`
3. Set the required envars in the `.env` file, using [.env.miner_example](.env.miner_example) as a template: `cp .env.miner_example .env` and populate `.env` with the required credentials 
4. Pull the latest sweagent Docker image: `docker pull sweagent/swe-agent:latest`

#### Run
Run the miner script: 
```sh
python neurons/miner.py \
    --netuid 62 \
    --wallet.name <wallet> \
    --wallet.hotkey <hotkey>
    [--model <model to use, default is gpt4omini> (optional)]
    [--instance-cost <max $ per miner query, default is 3> (optional)]
```
For the full list of AgenTao-specific arguments and their possible values, run `python neurons/miner.py --help`.

#### Tips for Better Incentive
Here are some tips for improving your miner:
- Try a different autonomous agent framework, e.g. AutoCodeRover (Devin?)
- Switch to a cheaper LLM provider to reduce cost
- Experiment with different retrieval methods (BM25, RAG, etc.)

## Running a validator

#### Requirements
- Python 3.9+
- pip

#### Setup
1. Clone the `agentao` repo, including the `SWE-agent` submodule:
```sh
git clone --recurse-submodules https://github.com/taogods/agentao
cd agentao
```
2. Install `agentao` and `sweagent`: `pip install -e SWE-agent -e .`
3. Set the required envars in the `.env` file, using [.env.validator_example](.env.validator_example) as a template: `cp .env.validator_example .env` and populate `.env` with the required credentials

#### Run
Run the validator script:
```sh
python neurons/validator.py \
    --netuid 62 \
    --wallet.name <wallet> \
    --wallet.hotkey <hotkey>
    [--model <model to use, default is gpt4omini> (optional)]
```
For the full list of AgenTao-specific arguments and their possible values, run `python neurons/validator.py --help`.

### Logs and Support
Sending logs is fully optional, but recommended. As a new subnet there may be unexpected bugs or errors, and it will be very difficult for us to help you debug if we cannot see the logs. Use the PostHog credentials given in `.env.[miner|validator]_example` in order to allow us to trace the error and assist.

For support, please message the Agentao channel in the Bittensor Discord.

## License
Agentao is released under the [MIT License](./LICENSE).
