import os
import urllib.request


def download_file(url, destination):
    """Download a file from a URL if it does not already exist."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        print(f"Downloading {url} to {destination}...")
        urllib.request.urlretrieve(url, destination)
        print("Download finished.")
    else:
        print(f"{destination} already exists.")
