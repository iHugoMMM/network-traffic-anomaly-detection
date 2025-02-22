# src/download_data.py

import os
import subprocess

def download_electricity_data():
    """
    Download the Electricity Load Diagrams dataset from Kaggle
    into data/raw/ (excluded from Git).
    """
    # Make sure data/raw exists
    os.makedirs('data/raw', exist_ok=True)

    # Kaggle CLI command
    cmd = [
        'kaggle', 'datasets', 'download',
        '-d', 'michaelrlooney/electricity-load-diagrams-2011-2014',
        '-p', 'data/raw'
    ]

    subprocess.run(cmd, check=True)
    print("Download complete. Unzipping...")

    # Unzip the file
    import zipfile
    zip_path = 'data/raw/electricity-load-diagrams-2011-2014.zip'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/raw')
    os.remove(zip_path)

    print("Unzipped and cleaned up zip file.")

if __name__ == '__main__':
    download_electricity_data()
