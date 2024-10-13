"""
Script to test the miner `generate_code_patch` function used in the miner forward pass.

Uses a sample issue from the SWE-agent tutorial:
https://princeton-nlp.github.io/SWE-agent/usage/cl_tutorial
"""
from taoception.miner_utils import UnsolvedIssue, generate_code_patch

test_issue_desc = """I'm running missing_colon.py as follows:

division(23, 0)
but I get the following error:

  File "/Users/fuchur/Documents/24/git_sync/swe-agent-test-repo/tests/./missing_colon.py", line 4
    def division(a: float, b: float) -> float
                                             ^
SyntaxError: invalid syntax
"""

test_unsolved_issue = UnsolvedIssue(
    desc=test_issue_desc,
    code_link="https://github.com/SWE-agent/test-repo",
)

patch = generate_code_patch(test_unsolved_issue).patch

# Breakpoint for debugging/analysis
import ipdb; ipdb.set_trace()
