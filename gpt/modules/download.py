import os
import sys
import requests
from tqdm import tqdm


def download_file_if_missing(url, save_path):
    """
    Downloads a file from a specified URL if it doesn't already exist at the given path.

    This function checks if the file already exists at the provided `save_path`. If the file is missing, it
    attempts to download the file from the specified `url` and saves it at the `save_path`. If the file already
    exists, it skips the download.

    Args:
        url (str): The URL from which to download the file.
        save_path (str): The local file path where the downloaded file should be saved.

    Returns:
        None: This function does not return any value. It directly downloads the file or skips the operation.

    Raises:
        SystemExit: If an error occurs during the file download, the program exits with an error message.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if the file already exists at the save path
    if not os.path.exists(save_path):
        print(f"File {save_path} not found. Downloading...")
        try:
            # Attempt to download the file from the URL
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise exception for HTTP errors
            file_size = int(response.headers.get('content-length', 0))

            # Use tqdm to show download progress
            with open(save_path, 'wb') as f, tqdm(
                    desc=f"Downloading {save_path}",
                    total=file_size,
                    unit='B',
                    unit_scale=True
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

            print(f"Downloaded {save_path}")
        except requests.exceptions.RequestException as e:
            # Print the error and exit if the download fails
            print(f"Error downloading the file: {e}")
            sys.exit(1)
    else:
        print(f"File {save_path} already exists. Skipping download.")
