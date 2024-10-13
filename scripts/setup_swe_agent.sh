#!/usr/bin/env bash
set -euxo pipefail

pushd .. > /dev/null

docker pull sweagent/swe-agent:latest

# Set up SWEAgent repo
git submodule update --init --recursive

python3 -m pip install --upgrade pip
pythom3 -m pip install --editable SWEAgent

popd > /dev/null
