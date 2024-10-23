from pathlib import Path
from botocore import UNSIGNED
from botocore.client import Config
import boto3


def download_repo_locally(s3_code_link: str, local_dir: str = None) -> Path:
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket_name, key = s3_code_link.replace("s3://", "").split('/', 1)

    # Set the local path where the file will be saved
    local_path_root = local_dir or Path.cwd()
    local_file_path = local_path_root / key.split('/')[-1]

    s3.Bucket(bucket_name).download_file(key, str(local_file_path))
    return local_file_path


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
