import os
import requests
from tqdm import tqdm

# URLs of the files to download
file_urls = [
    "https://storage.googleapis.com/ulascan/transformers-bert/config.json",
    "https://storage.googleapis.com/ulascan/transformers-bert/tf_model.h5"
]

# Directory to save the downloaded files
download_dir = "transformers-bert"

# Function to download a file from a URL with progress bar
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, tqdm(
                desc=save_path,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))
        print(f"Downloaded: {url} to {save_path}")
    else:
        print(f"Failed to download: {url}")

# Check if the directory exists and is not empty
if os.path.exists(download_dir) and os.listdir(download_dir):
    print(f"Directory '{download_dir}' exists and is not empty. Skipping download.")
else:
    # Create the directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Download each file to the specified directory
    for url in file_urls:
        file_name = os.path.basename(url)
        save_path = os.path.join(download_dir, file_name)
        download_file(url, save_path)
