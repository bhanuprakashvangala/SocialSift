import os
import urllib.request


def download_file(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {url} to {destination}...")
        urllib.request.urlretrieve(url, destination)
        print("Download finished.")
    else:
        print(f"{destination} already exists.")
