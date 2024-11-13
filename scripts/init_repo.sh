#!/usr/bin/env bash
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pushd $REPO_ROOT > /dev/null

git submodule update --init --recursive

python3 -m pip install --upgrade pip

pushd SWE-agent > /dev/null
python3 -m pip install --editable .
popd > /dev/null

python3 -m pip install torch
python3 -m pip install --editable .

docker pull sweagent/swe-agent:latest

popd > /dev/null
