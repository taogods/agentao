import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config


def download_repo_locally(s3_code_link: str, local_dir: Path | str = None) -> Path:
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket_name, key = s3_code_link.replace("s3://", "").split('/', 1)

    # Set the local path where the file will be saved
    local_path_root = Path(local_dir or Path.cwd())
    local_file_path = local_path_root / key.split('/')[-1]

    # Download the file
    s3.Bucket(bucket_name).download_file(key, str(local_file_path))

    # Check if it's a zip file
    if local_file_path.suffix == '.zip':
        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_extract_dir:
            temp_extract_path = Path(temp_extract_dir)

            # Extract the zip file into the temporary directory
            with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_path)

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
        for command in commands_to_run:
            subprocess.run(command.split(), shell=False)
        print(f"Initialized and committed in a Git repository at {target_dir}")
    finally:
        # Always return to the original directory
        os.chdir(original_dir)
        print(f"Returned to the original directory: {original_dir}")


def delete_s3_folder(s3_folder_link: str) -> None:
    # Remove the "s3://" part and split the bucket and key
    s3_folder_link = s3_folder_link.replace("s3://", "")
    bucket_name, folder_prefix = s3_folder_link.split('/', 1)

    # Create a boto3 resource for S3
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(bucket_name)

    # Delete all objects within the folder (prefix)
    bucket.objects.filter(Prefix=folder_prefix).delete()

    print(f"Deleted folder '{folder_prefix}' and all its contents from bucket '{bucket_name}'.")