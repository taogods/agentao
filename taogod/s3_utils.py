import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config

logger = logging.getLogger(__name__)

def download_repo_locally(s3_code_link: str, local_dir: Path | str = None) -> Path:
    logger.info(f"Entering download_repo_locally with URL {s3_code_link} and local_dir {local_dir}")

    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))

    bucket_name, key = s3_code_link.replace("s3://", "").split('/', 1)
    logger.debug(f"Using bucket name {bucket_name} and key {key}")

    # Set the local path where the file will be saved
    local_path_root = Path(local_dir or Path.cwd())
    local_file_path = local_path_root / key.split('/')[-1]

    logger.info(f"Downloading file from {bucket_name}/{key} to {local_file_path}...")
    s3.Bucket(bucket_name).download_file(key, str(local_file_path))
    logger.info(f"Finished downloading file to {local_file_path}")

    if local_file_path.suffix != '.zip':
        raise ValueError(f"Local file path {local_file_path} with suffix "
                         f"'{local_file_path.suffix} was not a .zip file")

    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_extract_dir:
        temp_extract_path = Path(temp_extract_dir)

        # Extract the zip file into the temporary directory
        logger.debug(f"Extracting zip file {local_file_path} into {temp_extract_path}...")
        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)
        logger.debug(f"Finished extracting zip file into {temp_extract_path}")

        # Find the one directory inside the extracted folder
        extracted_dir = next(temp_extract_path.iterdir())

        # Create the new folder name (same as zip file name without the extension)
        renamed_dir = local_path_root / local_file_path.stem

        # Rename the extracted directory if the target doesn't yet exist
        if not renamed_dir.exists():
            extracted_dir.rename(renamed_dir)
        _apply_git_commands(renamed_dir)

        # Delete the zip file after extraction
        os.remove(local_file_path)

        logger.info(f"Exiting download_repo_locally, returning {renamed_dir}")
        return renamed_dir  # Return the renamed directory path


# SWE-Agent requires that a git repo be initialized
def _apply_git_commands(target_dir: Path) -> None:
    original_dir = os.getcwd()

    commands_to_run = [
        "git init",
        "git add .",
        "git commit -m 'initial' --allow-empty",
    ]

    try:
        os.chdir(target_dir)
        logger.debug(f"Changed directories to {target_dir}")

        for command in commands_to_run:
            logger.debug(f"Running `{command}`...")
            subprocess.run(command.split(), shell=False)
            logger.debug(f"Finished running `{command}`")

        logger.info(f"Initialized and committed in a Git repository at {target_dir}")
    finally:
        # Always return to the original directory
        os.chdir(original_dir)
        logger.info(f"Returned to the original directory: {original_dir}")
